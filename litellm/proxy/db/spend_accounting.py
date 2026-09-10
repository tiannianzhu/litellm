from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import timedelta
from itertools import chain, groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, RootModel, TypeAdapter

from litellm.proxy.db.autorouter_session_rollup import AutoRouterTurnTransaction, write_autorouter_turn
from litellm.proxy.db.baseline_accounting import BaselineAccountingRecord, BaselineAccountingStore
from litellm.proxy.db.create_views import SupportsRawQueries
from litellm.proxy.db.daily_spend_bulk_upsert import (
    DAILY_SPEND_TABLES,
    DailySpendEntity,
    build_bulk_upsert,
    merge_by_conflict_key,
)
from litellm.proxy.db.model_usage_rollup import ModelUsageTransaction
from litellm.proxy.db.rollup_lock_timeout import ROLLUP_LOCK_TIMEOUT_SQL, rollup_lock_timeout_setting
from litellm.proxy.db.spend_log_tool_index import ToolUsageTransaction

if TYPE_CHECKING:
    from prisma import Prisma

    from litellm.proxy.utils import PrismaClient


class AccountingEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = 1
    event_key: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    call_type: str = Field(min_length=1)
    spend: float = Field(allow_inf_nan=False)
    payload_json: str = Field(min_length=1)


AccountingTransactionFactory: TypeAlias = Callable[[], AbstractAsyncContextManager[SupportsRawQueries]]
AccountingAccept: TypeAlias = Callable[[SupportsRawQueries, AccountingEvent], Awaitable[bool]]
AccountingApply: TypeAlias = Callable[[SupportsRawQueries, tuple[AccountingEvent, ...]], Awaitable[None]]


class EntityCharge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    entity: Literal[
        "key",
        "user",
        "end_user",
        "team",
        "team_member",
        "org",
        "org_member",
        "project",
        "tag",
        "model_access_group",
        "agent",
    ]
    entity_id: str
    spend: float
    user_id: str | None = None


class DailyCharge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    entity: DailySpendEntity
    row_json: str


class WindowCharge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    entity_type: Literal["key", "team"]
    entity_id: str
    window_duration: str
    window_start: str
    spend: float
    started_at: str | None = None


class AccountingEffects(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    charges: tuple[EntityCharge, ...]
    daily: tuple[DailyCharge, ...]
    windows: tuple[WindowCharge, ...] = ()
    model_usage: ModelUsageTransaction | None = None
    tools: ToolUsageTransaction | None = None
    autorouter: AutoRouterTurnTransaction | None = None
    baseline: BaselineAccountingRecord | None = None


class _Admission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    duplicate: bool
    succeeded: bool


class _TransactionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    statement_timeout: str
    lock_timeout: str


class _BoundTransaction:
    def __init__(self, db: SupportsRawQueries) -> None:
        self.db: Final = db

    async def __aenter__(self) -> SupportsRawQueries:
        return self.db

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> Literal[False]:
        return False


class _RawRow(RootModel[Mapping[str, object]]):
    model_config = ConfigDict(frozen=True, strict=True)


class _PrismaQueries:
    def __init__(self, db: Prisma) -> None:
        self.db: Final = db

    async def execute_raw(self, query: str, *args: object) -> int:
        return await self.db.execute_raw(query, *args)  # pyright: ignore[reportArgumentType]  # SQL builders return str

    async def query_raw(self, query: str, *args: object) -> tuple[Mapping[str, object], ...]:
        rows: Final = await self.db.query_raw(
            query,  # pyright: ignore[reportArgumentType]  # SQL builders return str
            *args,
            model=_RawRow,
        )
        return tuple(row.root for row in rows)


_ADMISSIONS: Final = TypeAdapter(tuple[_Admission, ...])
_EVENTS: Final = TypeAdapter(tuple[AccountingEvent, ...])
_SETTINGS: Final = TypeAdapter(tuple[_TransactionSettings, ...])
_SPEND: Final = TypeAdapter(float)
_TRANSACTION_TIMEOUT: Final = timedelta(seconds=60)
_LOCK_REQUEST: Final = "SELECT pg_advisory_xact_lock(hashtextextended($1::text, 0))"
_ADMISSION: Final = """
SELECT EXISTS (
    SELECT 1 FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1
) AS duplicate, EXISTS (
    SELECT 1 FROM "LiteLLM_SpendAccountingEvent" WHERE request_id = $2 AND status = 'success'
) AS succeeded
"""
_INSERT_EVENT: Final = """
INSERT INTO "LiteLLM_SpendAccountingEvent"
    (event_key, request_id, status, call_type, spend, version, payload_json)
VALUES ($1, $2, $3, $4, $5::double precision, $6::integer, $7)
"""
_PENDING_EVENTS: Final = """
SELECT version, event_key, request_id, status, call_type, spend, payload_json
FROM "LiteLLM_SpendAccountingEvent"
WHERE processed_at IS NULL
ORDER BY created_at, event_key
LIMIT $1::integer
FOR UPDATE SKIP LOCKED
"""
_COMPLETE_EVENTS: Final = """
UPDATE "LiteLLM_SpendAccountingEvent"
SET processed_at = CURRENT_TIMESTAMP, payload_json = NULL
WHERE event_key IN (SELECT jsonb_array_elements_text($1::jsonb)) AND processed_at IS NULL
"""


class SpendAccountingStore:
    def __init__(self, transaction_factory: AccountingTransactionFactory) -> None:
        self.transaction_factory: Final = transaction_factory

    @classmethod
    def from_client(cls, client: PrismaClient) -> SpendAccountingStore:
        @asynccontextmanager
        async def transaction() -> AsyncGenerator[SupportsRawQueries]:
            async with client.tx(timeout=_TRANSACTION_TIMEOUT) as db:
                yield _PrismaQueries(db)

        return cls(transaction)

    async def append(self, event: AccountingEvent, accept: AccountingAccept | None = None) -> bool:
        async with self.transaction_factory() as db:
            await db.execute_raw(ROLLUP_LOCK_TIMEOUT_SQL, rollup_lock_timeout_setting())
            await db.execute_raw(_LOCK_REQUEST, f"litellm_spend_accounting:{event.request_id}")
            admissions: Final = _ADMISSIONS.validate_python(
                tuple(await db.query_raw(_ADMISSION, event.event_key, event.request_id))
            )
            admission: Final = admissions[0]
            if admission.duplicate or (admission.succeeded and event.status == "failure" and event.spend == 0):
                return False
            if accept is not None and not await accept(db, event):
                return False
            await db.execute_raw(
                _INSERT_EVENT,
                event.event_key,
                event.request_id,
                event.status,
                event.call_type,
                event.spend,
                event.version,
                event.payload_json,
            )
        return True

    async def flush(self, apply: AccountingApply, limit: int = 20) -> int:
        if limit <= 0:
            return 0
        async with self.transaction_factory() as db:
            await db.execute_raw(ROLLUP_LOCK_TIMEOUT_SQL, rollup_lock_timeout_setting())
            events: Final = _EVENTS.validate_python(tuple(await db.query_raw(_PENDING_EVENTS, limit)))
            if not events:
                return 0
            await apply(db, events)
            await db.execute_raw(_COMPLETE_EVENTS, json.dumps(tuple(event.event_key for event in events)))
        return len(events)


_ROWS: Final = TypeAdapter(Mapping[str, object])
_ENTITY_TARGETS: Final = MappingProxyType(
    {
        "key": ("LiteLLM_VerificationToken", "token"),
        "user": ("LiteLLM_UserTable", "user_id"),
        "team": ("LiteLLM_TeamTable", "team_id"),
        "org": ("LiteLLM_OrganizationTable", "organization_id"),
        "project": ("LiteLLM_ProjectTable", "project_id"),
        "tag": ("LiteLLM_TagTable", "tag_name"),
        "model_access_group": ("LiteLLM_ModelAccessGroupBudgetTable", "access_group_name"),
        "agent": ("LiteLLM_AgentsTable", "agent_id"),
    }
)
_TEAM_LOCK: Final = "SELECT pg_advisory_xact_lock(hashtext($1)) IS NULL AS locked"
_TEAM_MEMBER: Final = """
INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, spend, total_spend)
SELECT $1, $2, $3::double precision, $3::double precision
WHERE EXISTS (
    SELECT 1 FROM "LiteLLM_TeamTable"
    WHERE team_id = $2 AND members_with_roles @> jsonb_build_array(jsonb_build_object('user_id', $1::text))
) OR EXISTS (
    SELECT 1 FROM "LiteLLM_TeamMembership" WHERE user_id = $1 AND team_id = $2
)
ON CONFLICT (user_id, team_id) DO UPDATE
SET spend = "LiteLLM_TeamMembership".spend + EXCLUDED.spend,
    total_spend = "LiteLLM_TeamMembership".total_spend + EXCLUDED.total_spend
"""
_END_USER: Final = """
INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, blocked)
VALUES ($1, $2::double precision, false)
ON CONFLICT (user_id) DO UPDATE SET spend = "LiteLLM_EndUserTable".spend + EXCLUDED.spend
"""
_ORG_MEMBER: Final = """
UPDATE "LiteLLM_OrganizationMembership"
SET spend = spend + $3::double precision, updated_at = CURRENT_TIMESTAMP
WHERE user_id = $1 AND organization_id = $2
"""
_WINDOW_EXISTS: Final = """
SELECT window_start::text AS window_start FROM "LiteLLM_BudgetWindowSpend"
WHERE entity_type = $1 AND entity_id = $2 AND window_duration = $3
"""
_WINDOW_UPSERT: Final = """
INSERT INTO "LiteLLM_BudgetWindowSpend"
    (entity_type, entity_id, window_duration, window_start, spend, created_at, updated_at)
VALUES ($1, $2, $3, ($4::timestamptz AT TIME ZONE 'UTC'), $5::double precision,
    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT (entity_type, entity_id, window_duration) DO UPDATE SET
    spend = CASE WHEN "LiteLLM_BudgetWindowSpend".window_start >= EXCLUDED.window_start
        THEN "LiteLLM_BudgetWindowSpend".spend + $6::double precision ELSE EXCLUDED.spend END,
    window_start = GREATEST("LiteLLM_BudgetWindowSpend".window_start, EXCLUDED.window_start),
    updated_at = CURRENT_TIMESTAMP
"""
_TOOL_INDEX: Final = """
INSERT INTO "LiteLLM_SpendLogToolIndex" (request_id, tool_name, start_time)
VALUES ($1, $2, ($3::timestamptz AT TIME ZONE 'UTC'))
ON CONFLICT (request_id, tool_name) DO NOTHING
"""
_TOOL_DAY: Final = """
INSERT INTO "LiteLLM_DailyToolSpend" (date, tool_name, spend, total_tokens, request_count, created_at, updated_at)
VALUES ($1, $2, $3::double precision, $4::bigint, $5::bigint, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT (date, tool_name) DO UPDATE SET
    spend = "LiteLLM_DailyToolSpend".spend + EXCLUDED.spend,
    total_tokens = "LiteLLM_DailyToolSpend".total_tokens + EXCLUDED.total_tokens,
    request_count = "LiteLLM_DailyToolSpend".request_count + EXCLUDED.request_count,
    updated_at = CURRENT_TIMESTAMP
"""
_MODEL_DAY: Final = """
INSERT INTO "LiteLLM_DailyModelUsage"
    (date, model_group, model, custom_llm_provider, task_type, spend, prompt_tokens, completion_tokens,
    request_count, successful_requests, failed_requests, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6::double precision, $7::bigint, $8::bigint, $9::bigint, $10::bigint,
    $11::bigint, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT (date, model_group, model, custom_llm_provider, task_type) DO UPDATE SET
    spend = "LiteLLM_DailyModelUsage".spend + EXCLUDED.spend,
    prompt_tokens = "LiteLLM_DailyModelUsage".prompt_tokens + EXCLUDED.prompt_tokens,
    completion_tokens = "LiteLLM_DailyModelUsage".completion_tokens + EXCLUDED.completion_tokens,
    request_count = "LiteLLM_DailyModelUsage".request_count + EXCLUDED.request_count,
    successful_requests = "LiteLLM_DailyModelUsage".successful_requests + EXCLUDED.successful_requests,
    failed_requests = "LiteLLM_DailyModelUsage".failed_requests + EXCLUDED.failed_requests,
    updated_at = CURRENT_TIMESTAMP
"""


def _charge_key(charge: EntityCharge) -> tuple[str, str, str]:
    return charge.entity, charge.entity_id, charge.user_id or ""


async def _apply_entities(db: SupportsRawQueries, charges: Sequence[EntityCharge]) -> None:
    for _key, grouped in groupby(sorted(charges, key=_charge_key), key=_charge_key):
        entries = tuple(grouped)
        charge = entries[0]
        spend = sum(entry.spend for entry in entries)
        if charge.entity == "end_user":
            await db.execute_raw(_END_USER, charge.entity_id, spend)
        elif charge.entity in ("team_member", "org_member"):
            if charge.user_id is None:
                raise ValueError("A membership charge requires user_id")
            await db.execute_raw(
                _TEAM_MEMBER if charge.entity == "team_member" else _ORG_MEMBER,
                charge.user_id,
                charge.entity_id,
                spend,
            )
        else:
            table, column = _ENTITY_TARGETS[charge.entity]
            extra = (
                ", total_spend = total_spend + $2::double precision, last_active = CURRENT_TIMESTAMP"
                if charge.entity == "key"
                else ""
            )
            await db.execute_raw(
                f'UPDATE "{table}" SET spend = spend + $2::double precision, updated_at = CURRENT_TIMESTAMP'
                f'{extra} WHERE "{column}" = $1',
                charge.entity_id,
                spend,
            )


async def _apply_daily(db: SupportsRawQueries, charges: Sequence[DailyCharge]) -> None:
    for entity, table in DAILY_SPEND_TABLES.items():
        rows = tuple(_ROWS.validate_json(charge.row_json) for charge in charges if charge.entity == entity)
        if rows:
            sql, params = build_bulk_upsert(table, merge_by_conflict_key(table, rows))
            await db.execute_raw(sql, *params)


def _window_key(charge: WindowCharge) -> tuple[str, str, str, str]:
    return charge.entity_type, charge.entity_id, charge.window_duration, charge.window_start


async def _apply_windows(db: SupportsRawQueries, charges: Sequence[WindowCharge]) -> None:
    for key, grouped in groupby(sorted(charges, key=_window_key), key=_window_key):
        entries = tuple(grouped)
        charge = entries[0]
        spend = sum(entry.spend for entry in entries)
        await db.execute_raw(_LOCK_REQUEST, "litellm_spend_window:" + json.dumps(key[:3]))
        existing = await db.query_raw(_WINDOW_EXISTS, *key[:3])
        column = "api_key" if charge.entity_type == "key" else "team_id"
        seed = (
            ()
            if existing
            else await db.query_raw(
                f'''SELECT COALESCE(SUM(log.spend), 0.0) AS spend FROM "LiteLLM_SpendLogs" log
WHERE log."{column}" = $1 AND log."startTime" >= ($2::timestamptz AT TIME ZONE 'UTC')
AND NOT EXISTS (
    SELECT 1 FROM "LiteLLM_SpendAccountingEvent" pending
    WHERE pending.request_id = log.request_id AND pending.status = COALESCE(log.status, 'success')
      AND pending.spend = log.spend AND pending.processed_at IS NULL
      AND EXISTS (
          SELECT 1 FROM jsonb_array_elements(pending.payload_json::jsonb -> 'windows') target
          WHERE target ->> 'entity_type' = $3 AND target ->> 'entity_id' = $1
            AND target ->> 'window_duration' = $4
      )
)''',
                charge.entity_id,
                charge.window_start,
                charge.entity_type,
                charge.window_duration,
            )
        )
        seed_spend = _ROWS.validate_python(seed[0]).get("spend", 0.0) if seed else 0.0
        baseline = _SPEND.validate_python(seed_spend)
        await db.execute_raw(_WINDOW_UPSERT, *key, baseline + spend, spend)


def _model_key(transaction: ModelUsageTransaction) -> tuple[str, str, str, str, str]:
    key: Final = transaction.key
    return key.date, key.model_group, key.model, key.custom_llm_provider, key.task_type


async def _apply_models(db: SupportsRawQueries, transactions: Sequence[ModelUsageTransaction]) -> None:
    for key, grouped in groupby(sorted(transactions, key=_model_key), key=_model_key):
        entries = tuple(grouped)
        successful = sum(entry.successful for entry in entries)
        await db.execute_raw(
            _MODEL_DAY,
            *key,
            sum(entry.spend for entry in entries),
            sum(entry.prompt_tokens for entry in entries),
            sum(entry.completion_tokens for entry in entries),
            len(entries),
            successful,
            len(entries) - successful,
        )


async def _apply_tools(db: SupportsRawQueries, transactions: Sequence[ToolUsageTransaction]) -> None:
    indexed: Final = tuple(
        chain.from_iterable(
            ((transaction.request_id, tool, transaction.start_time.isoformat()) for tool in transaction.tool_names)
            for transaction in transactions
        )
    )
    for row in sorted(indexed):
        await db.execute_raw(_TOOL_INDEX, *row)
    daily: Final = tuple(
        chain.from_iterable(
            ((transaction.date, tool, transaction.spend, transaction.total_tokens) for tool in transaction.tool_names)
            for transaction in transactions
        )
    )
    for key, grouped in groupby(sorted(daily), key=lambda row: row[:2]):
        entries = tuple(grouped)
        await db.execute_raw(
            _TOOL_DAY, *key, sum(row[2] for row in entries), sum(row[3] for row in entries), len(entries)
        )


async def _apply_baselines(db: SupportsRawQueries, records: Sequence[BaselineAccountingRecord]) -> None:
    if not records:
        return
    settings: Final = _SETTINGS.validate_python(
        tuple(
            await db.query_raw(
                "SELECT current_setting('statement_timeout') AS statement_timeout, "
                "current_setting('lock_timeout') AS lock_timeout"
            )
        )
    )[0]
    store: Final = BaselineAccountingStore(lambda: _BoundTransaction(db))
    for record in sorted(records, key=lambda item: (item.scope, item.observation.request_id)):
        if await store.append(record) == "unavailable":
            raise RuntimeError("Baseline accounting could not be applied")
    await db.execute_raw(
        "SELECT set_config('statement_timeout', $1::text, true), set_config('lock_timeout', $2::text, true)",
        settings.statement_timeout,
        settings.lock_timeout,
    )


async def apply_accounting_events(db: SupportsRawQueries, events: tuple[AccountingEvent, ...]) -> None:
    effects: Final = tuple(AccountingEffects.model_validate_json(event.payload_json) for event in events)
    charges: Final = tuple(chain.from_iterable(effect.charges for effect in effects))
    for team_id in sorted(frozenset(charge.entity_id for charge in charges if charge.entity == "team_member")):
        await db.execute_raw(_TEAM_LOCK, team_id)
    await _apply_baselines(db, tuple(effect.baseline for effect in effects if effect.baseline is not None))
    await _apply_entities(db, charges)
    await _apply_daily(db, tuple(chain.from_iterable(effect.daily for effect in effects)))
    await _apply_windows(db, tuple(chain.from_iterable(effect.windows for effect in effects)))
    await _apply_models(db, tuple(effect.model_usage for effect in effects if effect.model_usage is not None))
    await _apply_tools(db, tuple(effect.tools for effect in effects if effect.tools is not None))
    turns: Final = tuple(
        effect.autorouter for effect in effects if effect.autorouter is not None and effect.baseline is None
    )
    for turn in sorted(
        turns, key=lambda item: (item.api_key, item.user_id, item.session_id, item.router_name, item.turn_at)
    ):
        await write_autorouter_turn(db, turn)
