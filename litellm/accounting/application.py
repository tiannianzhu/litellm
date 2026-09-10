import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from pydantic import TypeAdapter

from litellm.accounting.delivery_projection import entity_transactions, project_deliveries
from litellm.accounting.projection import Contribution
from litellm.proxy.db.autorouter_session_rollup import AutoRouterTurnTransaction, write_autorouter_turn
from litellm.proxy.db.baseline_accounting import BaselineAccountingRecord, BaselineAccountingStore
from litellm.proxy.db.budget_window_spend_writer import commit_window_spend_updates
from litellm.proxy.db.daily_spend_bulk_upsert import DAILY_SPEND_TABLES, build_bulk_upsert, merge_by_conflict_key
from litellm.proxy.db.db_span import db_span
from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import WindowSpendTransaction
from litellm.proxy.db.model_usage_rollup import queue_model_usage_upsert
from litellm.proxy.db.rollup_lock_timeout import apply_rollup_lock_timeout
from litellm.proxy.db.spend_log_tool_index import ToolUsageTransaction
from litellm.repositories.accounting_delivery_repository import DeliveryProjection, LedgerTransaction
from litellm.repositories.table_repositories import SpendLogToolIndexRepository

if TYPE_CHECKING:
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    from litellm.proxy.utils import PrismaClient, ProxyLogging


_FIELDS: Final = TypeAdapter(dict[str, object])
_TOOLS: Final = TypeAdapter(ToolUsageTransaction)
_TURNS: Final = TypeAdapter(AutoRouterTurnTransaction)
_SETTINGS: Final = TypeAdapter(tuple[dict[str, str], ...])


@dataclass(frozen=True, slots=True)
class _BorrowedTransaction:
    transaction: LedgerTransaction

    async def __aenter__(self) -> LedgerTransaction:
        return self.transaction

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> Literal[False]:
        return False


async def apply_deliveries(
    transaction: LedgerTransaction,
    deliveries: tuple[DeliveryProjection, ...],
    *,
    prisma_client: "PrismaClient",
    writer: "DBSpendUpdateWriter",
    proxy_logging_obj: "ProxyLogging",
) -> tuple[Contribution, ...]:
    from litellm.proxy.db.db_spend_update_writer import _TEAM_ADVISORY_LOCK_SQL
    from litellm.proxy.guardrails.usage_tracking import UsageUnitIncrement, UsageUnitKey, upsert_usage_unit_row

    contributions: Final = project_deliveries(deliveries)
    metadata: Final = tuple(
        _FIELDS.validate_json(value) if isinstance(value, str) else _FIELDS.validate_python(value)
        for delivery in deliveries
        if (value := delivery.row.get("metadata", {})) is not None
    )
    baselines: Final = tuple(
        BaselineAccountingRecord.model_validate_json(json.dumps(record))
        for item in metadata
        if (record := item.get("accounting_baseline")) is not None
    )
    await apply_rollup_lock_timeout(transaction)
    for team_id in sorted(
        frozenset(dict(item.key)["team_id"] for item in contributions if item.table == "LiteLLM_TeamMembership")
    ):
        await transaction.execute_raw(_TEAM_ADVISORY_LOCK_SQL, team_id)
    await _apply_baselines(transaction, baselines)
    await writer.apply_accounting_entities(
        prisma_client, proxy_logging_obj, transaction, entity_transactions(contributions)
    )
    for table in sorted(DAILY_SPEND_TABLES.values(), key=lambda item: item.name):
        rows = tuple(
            {**dict(item.key), **dict(item.values), **dict(item.increments)}
            for item in contributions
            if item.table == table.name
        )
        merged = merge_by_conflict_key(table, rows)
        for offset in range(0, len(merged), 100):
            sql, params = build_bulk_upsert(table, merged[offset : offset + 100])
            async with db_span("upsert_daily_spend", table.name):
                await transaction.execute_raw(sql, *params)
    async with transaction.batch_() as batcher:
        for item in contributions:
            if item.table == "LiteLLM_DailyModelUsage":
                queue_model_usage_upsert(batcher, dict(item.key), dict(item.increments))
    for item in contributions:
        if item.table == "LiteLLM_DailyGuardrailUsageUnits":
            key = dict(item.key)
            counters = dict(item.increments)
            await upsert_usage_unit_row(
                prisma_client,
                UsageUnitKey(key["guardrail_id"], key["date"], key["team_id"], key["api_key"], key["usage_unit"]),
                UsageUnitIncrement(int(counters["units"]), float(counters["cost"]), int(counters["untracked_units"])),
                outer_transaction=transaction,
            )
    windows: Final = tuple(_window(item) for item in contributions if item.table == "LiteLLM_BudgetWindowSpend")
    await commit_window_spend_updates(prisma_client, windows, outer_transaction=transaction)
    await SpendLogToolIndexRepository.apply_transactions(
        transaction,
        tuple(
            _TOOLS.validate_python(tools) for item in metadata if (tools := item.get("accounting_tools")) is not None
        ),
    )
    turns: Final = tuple(
        _TURNS.validate_python(turn) for item in metadata if (turn := item.get("accounting_autorouter")) is not None
    )
    for turn in sorted(
        turns, key=lambda item: (item.api_key, item.user_id, item.session_id, item.router_name, item.turn_at)
    ):
        await write_autorouter_turn(transaction, turn)
    return contributions


async def _apply_baselines(transaction: LedgerTransaction, records: tuple[BaselineAccountingRecord, ...]) -> None:
    if not records:
        return
    settings: Final = _SETTINGS.validate_python(
        await transaction.query_raw(
            "SELECT current_setting('statement_timeout') AS statement_timeout, "
            "current_setting('lock_timeout') AS lock_timeout"
        )
    )[0]
    store: Final = BaselineAccountingStore(lambda: _BorrowedTransaction(transaction))
    for record in sorted(records, key=lambda item: (item.scope, item.observation.request_id)):
        if await store.append(record) == "unavailable":
            raise RuntimeError("Baseline accounting could not be applied")
    await transaction.execute_raw(
        "SELECT set_config('statement_timeout', $1::text, true), set_config('lock_timeout', $2::text, true)",
        settings["statement_timeout"],
        settings["lock_timeout"],
    )


def _window(item: Contribution) -> WindowSpendTransaction:
    key: Final = dict(item.key)
    values: Final = dict(item.values)
    return WindowSpendTransaction(
        entity_type=key["entity_type"],
        entity_id=key["entity_id"],
        window_duration=key["window_duration"],
        window_start=values["window_start"] or "",
        spend=float(dict(item.increments)["spend"]),
        started_at=values["started_at"],
    )
