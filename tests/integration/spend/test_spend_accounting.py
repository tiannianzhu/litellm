import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Final

import pytest
import pytest_asyncio
from prisma import Prisma

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.create_views import SupportsRawQueries
from litellm.proxy.db.db_spend_update_writer import _accept_accounting_spend_log
from litellm.proxy.db.model_usage_rollup import ModelUsageKey, ModelUsageTransaction
from litellm.proxy.db.spend_accounting import (
    AccountingEffects,
    AccountingEvent,
    DailyCharge,
    EntityCharge,
    SpendAccountingStore,
    WindowCharge,
    apply_accounting_events,
)
from litellm.proxy.db.spend_log_tool_index import ToolUsageTransaction
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.types.utils import CallTypes
from tests.integration._support.prisma_clients import connected_prisma

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest_asyncio.fixture(scope="module", loop_scope="session")
async def db() -> AsyncIterator[Prisma]:
    async with connected_prisma() as client:
        yield client


@asynccontextmanager
async def _transaction(
    db: Prisma, *, before_commit: bool = False, after_commit: bool = False
) -> AsyncIterator[SupportsRawQueries]:
    async with db.tx() as tx:
        yield tx
        if before_commit:
            raise RuntimeError("injected pre-commit interruption")
    if after_commit:
        raise RuntimeError("injected lost commit acknowledgement")


def _store(db: Prisma, *, before_commit: bool = False, after_commit: bool = False) -> SpendAccountingStore:
    def transaction():
        return _transaction(db, before_commit=before_commit, after_commit=after_commit)

    return SpendAccountingStore(transaction)


@pytest_asyncio.fixture
async def event(db: Prisma) -> AsyncIterator[Callable[..., AccountingEvent]]:
    run: Final = uuid.uuid4().hex

    def create(
        label: str = "event",
        request_id: str | None = None,
        status: str = "success",
        spend: float = 0.25,
        payload_json: str = '{"charges":[],"daily":[]}',
        call_type: str = "chat",
    ) -> AccountingEvent:
        return AccountingEvent(
            version=1,
            event_key=f"{run}:{label}",
            request_id=request_id or f"{run}:{label}",
            status=status,
            call_type=call_type,
            spend=spend,
            payload_json=payload_json,
        )

    try:
        yield create
    finally:
        await db.execute_raw(
            'DELETE FROM "LiteLLM_SpendAccountingEvent" WHERE event_key LIKE $1', f"{run}:%"
        )


@pytest_asyncio.fixture
async def ledger(db: Prisma) -> AsyncIterator[str]:
    table: Final = f"SpendAccountingContract_{uuid.uuid4().hex}"
    await db.execute_raw(
        f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY, spend DOUBLE PRECISION NOT NULL, requests INTEGER NOT NULL)'
    )
    await db.execute_raw(f'INSERT INTO "{table}" VALUES (1, 0, 0)')
    try:
        yield table
    finally:
        await db.execute_raw(f'DROP TABLE IF EXISTS "{table}"')


async def _apply(db: SupportsRawQueries, table: str, events: tuple[AccountingEvent, ...]) -> None:
    for item in events:
        await db.execute_raw(
            f'UPDATE "{table}" SET spend = spend + $1, requests = requests + 1 WHERE id = 1', item.spend
        )


async def _ledger(db: Prisma, table: str) -> tuple[float, int]:
    rows: Final = await db.query_raw(f'SELECT spend, requests FROM "{table}" WHERE id = 1')
    return float(rows[0]["spend"]), int(rows[0]["requests"])


async def _event_count(db: Prisma, keys: tuple[str, ...]) -> int:
    rows: Final = await db.query_raw(
        'SELECT COUNT(*) AS count FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = ANY($1::text[])', keys
    )
    return int(rows[0]["count"])


def _daily_row(entity_column: str, entity_id: str, api_key: str, day: str) -> str:
    metrics: Final = {
        "prompt_tokens": 11,
        "completion_tokens": 12,
        "cache_read_input_tokens": 13,
        "cache_creation_input_tokens": 14,
        "compression_saved_tokens": 15,
        "spend": 16.25,
        "compression_savings_spend": 17.25,
        "prompt_caching_savings_spend": 18.25,
        "gateway_injected_caching_savings_spend": 19.25,
        "autorouter_savings_spend": 20.25,
        "api_requests": 21,
        "successful_requests": 22,
        "failed_requests": 23,
        "total_response_time_ms": 24,
        "timed_requests": 25,
    }
    return json.dumps(
        {
            entity_column: entity_id,
            "date": day,
            "api_key": api_key,
            "model": f"contract-model-{api_key}",
            "model_group": f"contract-group-{api_key}",
            "custom_llm_provider": "contract-provider",
            "endpoint": "chat",
            **metrics,
        }
    )


def _daily_metrics() -> tuple[str, ...]:
    return (
        "prompt_tokens",
        "completion_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "compression_saved_tokens",
        "spend",
        "compression_savings_spend",
        "prompt_caching_savings_spend",
        "gateway_injected_caching_savings_spend",
        "autorouter_savings_spend",
        "api_requests",
        "successful_requests",
        "failed_requests",
        "total_response_time_ms",
        "timed_requests",
    )


async def _daily_values(
    db: Prisma, table: str, entity_column: str, entity_id: str, day: str, model: str
) -> tuple[object, ...]:
    columns: Final = ", ".join('"' + name + '"' for name in _daily_metrics())
    rows: Final = await db.query_raw(
        f'SELECT {columns} FROM "{table}" WHERE "{entity_column}" = $1 AND date = $2 AND model = $3',
        entity_id,
        day,
        model,
    )
    if not rows:
        return ()
    return tuple(rows[0][metric] for metric in _daily_metrics())


async def _global_daily_values(db: Prisma, day: str, model: str) -> tuple[object, ...]:
    columns: Final = ", ".join('"' + name + '"' for name in _daily_metrics())
    rows: Final = await db.query_raw(
        f'SELECT {columns} FROM "LiteLLM_DailyGlobalSpend" WHERE date = $1 AND model = $2', day, model
    )
    if not rows:
        return ()
    return tuple(rows[0][metric] for metric in _daily_metrics())


async def _entity_spend(db: Prisma, table: str, column: str, identity: str) -> float:
    rows: Final = await db.query_raw(f'SELECT spend FROM "{table}" WHERE "{column}" = $1', identity)
    return float(rows[0]["spend"])


def _writer_kwargs(user_id: str, tag: str) -> dict[str, object]:
    return {
        "model": "gpt-4o-mini",
        "call_type": CallTypes.acompletion.value,
        "litellm_params": {
            "metadata": {"user_api_key_user_id": user_id, "tags": [tag], "status": "success"}
        },
    }


async def test_append_is_deduplicated_across_store_recreation(
    db: Prisma, event: Callable[..., AccountingEvent]
) -> None:
    item: Final = event()
    assert await _store(db).append(item) is True
    assert await _store(db).append(item) is False
    assert await _event_count(db, (item.event_key,)) == 1


async def test_acceptance_and_event_commit_atomically(
    db: Prisma, event: Callable[..., AccountingEvent]
) -> None:
    item: Final = event()
    table: Final = f"SpendAccept_{uuid.uuid4().hex}"
    await db.execute_raw(
        f'CREATE TABLE "{table}" (request_id TEXT PRIMARY KEY)'
    )

    async def accept(tx: SupportsRawQueries, candidate: AccountingEvent) -> bool:
        inserted: Final = await tx.query_raw(
            f'INSERT INTO "{table}" VALUES ($1) ON CONFLICT DO NOTHING RETURNING request_id', candidate.request_id
        )
        return bool(inserted)

    try:
        with pytest.raises(RuntimeError, match="pre-commit"):
            await _store(db, before_commit=True).append(item, accept=accept)
        assert await db.query_raw(f'SELECT request_id FROM "{table}" WHERE request_id = $1', item.request_id) == []
        assert await _event_count(db, (item.event_key,)) == 0
        assert await _store(db).append(item, accept=accept) is True
        assert len(await db.query_raw(f'SELECT request_id FROM "{table}" WHERE request_id = $1', item.request_id)) == 1
        assert await _event_count(db, (item.event_key,)) == 1
    finally:
        await db.execute_raw(f'DROP TABLE IF EXISTS "{table}"')


async def test_flush_failure_rolls_back_then_replays(
    db: Prisma, ledger: str, event: Callable[..., AccountingEvent]
) -> None:
    item: Final = event()
    store: Final = _store(db)
    await store.append(item)

    async def fail_before_commit(tx: SupportsRawQueries, batch: tuple[AccountingEvent, ...]) -> None:
        await _apply(tx, ledger, batch)
        raise RuntimeError("apply failed before commit")

    with pytest.raises(RuntimeError, match="before commit"):
        await store.flush(fail_before_commit)
    assert await _ledger(db, ledger) == (0.0, 0)

    async def replay(tx: SupportsRawQueries, batch: tuple[AccountingEvent, ...]) -> None:
        await _apply(tx, ledger, batch)

    assert await _store(db).flush(replay) == 1
    assert await _ledger(db, ledger) == pytest.approx((item.spend, 1))


@pytest.mark.parametrize("failure_phase", ("before_commit", "after_commit"))
async def test_flush_recovery_preserves_exactly_once_ledger(
    db: Prisma, ledger: str, event: Callable[..., AccountingEvent], failure_phase: str
) -> None:
    item: Final = event()
    await _store(db).append(item)

    async def apply(tx: SupportsRawQueries, batch: tuple[AccountingEvent, ...]) -> None:
        await _apply(tx, ledger, batch)

    faulted: Final = _store(
        db,
        before_commit=failure_phase == "before_commit",
        after_commit=failure_phase == "after_commit",
    )
    if failure_phase == "before_commit":
        with pytest.raises(RuntimeError, match="pre-commit"):
            await faulted.flush(apply)
        assert await _ledger(db, ledger) == (0.0, 0)
    else:
        with pytest.raises(RuntimeError, match="lost commit"):
            await faulted.flush(apply)
        assert await _ledger(db, ledger) == pytest.approx((item.spend, 1))

    assert await _store(db).flush(apply) == (1 if failure_phase == "before_commit" else 0)
    assert await _ledger(db, ledger) == pytest.approx((item.spend, 1))


async def test_concurrent_flushes_and_request_outcomes_are_accounted_once(
    db: Prisma, ledger: str, event: Callable[..., AccountingEvent]
) -> None:
    accepted: Final = event("accepted", request_id="shared-request")
    free_failure: Final = event("free-failure", request_id="shared-request", status="failure", spend=0)
    billable_failure: Final = event("billable-failure", request_id="shared-request", status="failure", spend=0.1)
    separate_request: Final = event("separate", request_id="other-request", spend=0.2)
    store: Final = _store(db)

    async def accept(tx: SupportsRawQueries, candidate: AccountingEvent) -> bool:
        return True

    assert await store.append(accepted, accept=accept) is True
    assert await store.append(free_failure, accept=accept) is False
    assert await store.append(billable_failure, accept=accept) is True
    assert await store.append(separate_request, accept=accept) is True

    async def apply(tx: SupportsRawQueries, batch: tuple[AccountingEvent, ...]) -> None:
        await _apply(tx, ledger, batch)

    processed: Final = await asyncio.gather(
        _store(db).flush(apply), _store(db).flush(apply)
    )
    assert sum(processed) == 3
    assert await _ledger(db, ledger) == pytest.approx((0.55, 3))
    assert await _store(db).flush(apply) == 0


async def test_six_daily_entities_and_user_global_metrics_commit_together_and_replay(
    db: Prisma, event: Callable[..., AccountingEvent]
) -> None:
    run: Final = uuid.uuid4().hex
    day: Final = "2199-01-01"
    api_key: Final = f"daily-key-{run}"
    user_id: Final = f"daily-user-{run}"
    model: Final = f"contract-model-{api_key}"
    daily_targets: Final = (
        ("user", "LiteLLM_DailyUserSpend", "user_id", user_id),
        ("team", "LiteLLM_DailyTeamSpend", "team_id", f"daily-team-{run}"),
        ("org", "LiteLLM_DailyOrganizationSpend", "organization_id", f"daily-org-{run}"),
        ("end_user", "LiteLLM_DailyEndUserSpend", "end_user_id", f"daily-end-user-{run}"),
        ("tag", "LiteLLM_DailyTagSpend", "tag", f"daily-tag-{run}"),
        ("agent", "LiteLLM_DailyAgentSpend", "agent_id", f"daily-agent-{run}"),
    )
    effects: Final = AccountingEffects(
        charges=(),
        daily=tuple(
            DailyCharge(entity=entity, row_json=_daily_row(entity_column, identity, api_key, day))
            for entity, _table, entity_column, identity in daily_targets
        ),
    )
    item: Final = event("daily", payload_json=effects.model_dump_json())
    try:
        await _store(db).append(item)
        with pytest.raises(RuntimeError, match="pre-commit"):
            await _store(db, before_commit=True).flush(apply_accounting_events)
        for _entity, table, entity_column, identity in daily_targets:
            assert await _daily_values(db, table, entity_column, identity, day, model) == ()
        assert await _global_daily_values(db, day, model) == ()

        assert await _store(db).flush(apply_accounting_events) == 1
        expected: Final = (11, 12, 13, 14, 15, 16.25, 17.25, 18.25, 19.25, 20.25, 21, 22, 23, 24, 25)
        for _entity, table, entity_column, identity in daily_targets:
            assert await _daily_values(db, table, entity_column, identity, day, model) == pytest.approx(expected)
        assert await _global_daily_values(db, day, model) == pytest.approx(expected)
    finally:
        for _entity, table, entity_column, identity in daily_targets:
            await db.execute_raw(
                f'DELETE FROM "{table}" WHERE "{entity_column}" = $1 AND date = $2 AND model = $3',
                identity,
                day,
                model,
            )
        await db.execute_raw(
            'DELETE FROM "LiteLLM_DailyGlobalSpend" WHERE date = $1 AND model = $2', day, model
        )


async def test_entity_and_membership_balances_apply_from_durable_event(
    db: Prisma, event: Callable[..., AccountingEvent]
) -> None:
    run: Final = uuid.uuid4().hex
    user_id: Final = f"acct-user-{run}"
    team_id: Final = f"acct-team-{run}"
    org_id: Final = f"acct-org-{run}"
    key: Final = f"acct-key-{run}"
    end_user: Final = f"acct-end-user-{run}"
    project: Final = f"acct-project-{run}"
    tag: Final = f"acct-tag-{run}"
    group: Final = f"acct-group-{run}"
    agent: Final = f"acct-agent-{run}"
    budget: Final = f"acct-budget-{run}"
    await db.execute_raw(
        'INSERT INTO "LiteLLM_BudgetTable" (budget_id, created_by, updated_by) VALUES ($1, $2, $2)',
        budget,
        user_id,
    )
    await db.execute_raw(
        'INSERT INTO "LiteLLM_OrganizationTable" '
        '(organization_id, organization_alias, budget_id, models, created_by, updated_by) '
        "VALUES ($1, $1, $2, ARRAY[]::text[], $3, $3)",
        org_id,
        budget,
        user_id,
    )
    await db.execute_raw(
        'INSERT INTO "LiteLLM_UserTable" (user_id, models, organization_id) VALUES ($1, ARRAY[]::text[], $2)',
        user_id,
        org_id,
    )
    await db.execute_raw(
        'INSERT INTO "LiteLLM_TeamTable" (team_id, admins, members, models) '
        'VALUES ($1, ARRAY[]::text[], ARRAY[]::text[], ARRAY[]::text[])',
        team_id,
    )
    await db.execute_raw(
        'INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id) VALUES ($1, $2)', user_id, team_id
    )
    await db.execute_raw(
        'INSERT INTO "LiteLLM_OrganizationMembership" (user_id, organization_id) VALUES ($1, $2)',
        user_id,
        org_id,
    )
    await db.execute_raw(
        'INSERT INTO "LiteLLM_VerificationToken" (token, models) VALUES ($1, ARRAY[]::text[])', key
    )
    await db.execute_raw('INSERT INTO "LiteLLM_EndUserTable" (user_id) VALUES ($1)', end_user)
    await db.execute_raw(
        'INSERT INTO "LiteLLM_ProjectTable" (project_id, models, created_by, updated_by) '
        'VALUES ($1, ARRAY[]::text[], $2, $2)',
        project,
        user_id,
    )
    await db.execute_raw(
        'INSERT INTO "LiteLLM_TagTable" (tag_name, models) VALUES ($1, ARRAY[]::text[])', tag
    )
    await db.execute_raw(
        'INSERT INTO "LiteLLM_ModelAccessGroupBudgetTable" (access_group_name) VALUES ($1)', group
    )
    await db.execute_raw(
        'INSERT INTO "LiteLLM_AgentsTable" '
        '(agent_id, agent_name, agent_card_params, created_by, updated_by) VALUES ($1, $1, $2::jsonb, $3, $3)',
        agent,
        "{}",
        user_id,
    )
    effects: Final = AccountingEffects(
        charges=(
            EntityCharge(entity="key", entity_id=key, spend=0.5),
            EntityCharge(entity="user", entity_id=user_id, spend=0.5),
            EntityCharge(entity="end_user", entity_id=end_user, spend=0.5),
            EntityCharge(entity="team", entity_id=team_id, spend=0.5),
            EntityCharge(entity="team_member", entity_id=team_id, user_id=user_id, spend=0.5),
            EntityCharge(entity="org", entity_id=org_id, spend=0.5),
            EntityCharge(entity="org_member", entity_id=org_id, user_id=user_id, spend=0.5),
            EntityCharge(entity="project", entity_id=project, spend=0.5),
            EntityCharge(entity="tag", entity_id=tag, spend=0.5),
            EntityCharge(entity="model_access_group", entity_id=group, spend=0.5),
            EntityCharge(entity="agent", entity_id=agent, spend=0.5),
        ),
        daily=(),
    )
    item: Final = event("entities", payload_json=effects.model_dump_json())
    try:
        await _store(db).append(item)
        assert await _store(db).flush(apply_accounting_events) == 1
        targets: Final = (
            ("LiteLLM_VerificationToken", "token", key),
            ("LiteLLM_UserTable", "user_id", user_id),
            ("LiteLLM_EndUserTable", "user_id", end_user),
            ("LiteLLM_TeamTable", "team_id", team_id),
            ("LiteLLM_OrganizationTable", "organization_id", org_id),
            ("LiteLLM_ProjectTable", "project_id", project),
            ("LiteLLM_TagTable", "tag_name", tag),
            ("LiteLLM_ModelAccessGroupBudgetTable", "access_group_name", group),
            ("LiteLLM_AgentsTable", "agent_id", agent),
            ("LiteLLM_TeamMembership", "team_id", team_id),
            ("LiteLLM_OrganizationMembership", "organization_id", org_id),
        )
        for table, column, identity in targets:
            assert await _entity_spend(db, table, column, identity) == pytest.approx(0.5)
        key_rows: Final = await db.query_raw(
            'SELECT total_spend FROM "LiteLLM_VerificationToken" WHERE token = $1', key
        )
        assert float(key_rows[0]["total_spend"]) == pytest.approx(0.5)
        member_rows: Final = await db.query_raw(
            'SELECT spend, total_spend FROM "LiteLLM_TeamMembership" WHERE user_id = $1 AND team_id = $2',
            user_id,
            team_id,
        )
        assert (float(member_rows[0]["spend"]), float(member_rows[0]["total_spend"])) == pytest.approx((0.5, 0.5))
    finally:
        await db.execute_raw('DELETE FROM "LiteLLM_TeamMembership" WHERE user_id = $1', user_id)
        await db.execute_raw('DELETE FROM "LiteLLM_OrganizationMembership" WHERE user_id = $1', user_id)
        await db.execute_raw('DELETE FROM "LiteLLM_AgentsTable" WHERE agent_id = $1', agent)
        await db.execute_raw('DELETE FROM "LiteLLM_ModelAccessGroupBudgetTable" WHERE access_group_name = $1', group)
        await db.execute_raw('DELETE FROM "LiteLLM_TagTable" WHERE tag_name = $1', tag)
        await db.execute_raw('DELETE FROM "LiteLLM_ProjectTable" WHERE project_id = $1', project)
        await db.execute_raw('DELETE FROM "LiteLLM_EndUserTable" WHERE user_id = $1', end_user)
        await db.execute_raw('DELETE FROM "LiteLLM_VerificationToken" WHERE token = $1', key)
        await db.execute_raw('DELETE FROM "LiteLLM_TeamTable" WHERE team_id = $1', team_id)
        await db.execute_raw('DELETE FROM "LiteLLM_UserTable" WHERE user_id = $1', user_id)
        await db.execute_raw('DELETE FROM "LiteLLM_OrganizationTable" WHERE organization_id = $1', org_id)
        await db.execute_raw('DELETE FROM "LiteLLM_BudgetTable" WHERE budget_id = $1', budget)


async def test_model_usage_and_tool_index_rollups_apply_from_same_event(
    db: Prisma, event: Callable[..., AccountingEvent]
) -> None:
    run: Final = uuid.uuid4().hex
    day: Final = "2199-02-03"
    request_id: Final = f"usage-request-{run}"
    model_group: Final = f"usage-group-{run}"
    model: Final = f"usage-model-{run}"
    tool_name: Final = f"usage-tool-{run}"
    usage: Final = ModelUsageTransaction(
        key=ModelUsageKey(day, model_group, model, "contract-provider", "contract-task"),
        spend=0.3,
        prompt_tokens=4,
        completion_tokens=5,
        successful=True,
    )
    tools: Final = ToolUsageTransaction(
        request_id=request_id,
        date=day,
        start_time=datetime.fromisoformat(f"{day}T12:00:00+00:00"),
        tool_names=(tool_name,),
        spend=0.3,
        total_tokens=9,
    )
    effects: Final = AccountingEffects(charges=(), daily=(), model_usage=usage, tools=tools)
    try:
        await _store(db).append(event("usage", request_id=request_id, payload_json=effects.model_dump_json()))
        assert await _store(db).flush(apply_accounting_events) == 1

        model_rows: Final = await db.query_raw(
            'SELECT spend, prompt_tokens, completion_tokens, request_count, successful_requests, failed_requests '
            'FROM "LiteLLM_DailyModelUsage" WHERE date = $1 AND model_group = $2 AND model = $3',
            day,
            model_group,
            model,
        )
        assert tuple(
            model_rows[0][column]
            for column in (
                "spend",
                "prompt_tokens",
                "completion_tokens",
                "request_count",
                "successful_requests",
                "failed_requests",
            )
        ) == pytest.approx((0.3, 4, 5, 1, 1, 0))
        index_rows: Final = await db.query_raw(
            'SELECT tool_name FROM "LiteLLM_SpendLogToolIndex" WHERE request_id = $1', request_id
        )
        assert tuple(row["tool_name"] for row in index_rows) == (tool_name,)
        tool_rows: Final = await db.query_raw(
            'SELECT spend, total_tokens, request_count FROM "LiteLLM_DailyToolSpend" '
            'WHERE date = $1 AND tool_name = $2',
            day,
            tool_name,
        )
        assert tuple(
            tool_rows[0][column] for column in ("spend", "total_tokens", "request_count")
        ) == pytest.approx((0.3, 9, 1))
    finally:
        await db.execute_raw(
            'DELETE FROM "LiteLLM_DailyModelUsage" WHERE date = $1 AND model_group = $2 AND model = $3',
            day,
            model_group,
            model,
        )
        await db.execute_raw('DELETE FROM "LiteLLM_SpendLogToolIndex" WHERE request_id = $1', request_id)
        await db.execute_raw(
            'DELETE FROM "LiteLLM_DailyToolSpend" WHERE date = $1 AND tool_name = $2', day, tool_name
        )


async def test_window_seed_excludes_pending_event_and_charges_late_old_window_once(
    db: Prisma, event: Callable[..., AccountingEvent]
) -> None:
    run: Final = uuid.uuid4().hex
    api_key: Final = f"window-key-{run}"
    legacy_request: Final = f"window-legacy-{run}"
    current_request: Final = f"window-current-{run}"
    old_start: Final = "2199-03-01T00:00:00+00:00"
    new_start: Final = "2199-03-02T00:00:00+00:00"
    await db.execute_raw(
        'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, api_key, spend, "startTime", "endTime", status) '
        'VALUES ($1, $2, $3, $4, $5::timestamptz, $5::timestamptz, $6)',
        legacy_request,
        "chat",
        api_key,
        0.2,
        old_start,
        "success",
    )
    try:
        first_effects: Final = AccountingEffects(
            charges=(),
            daily=(),
            windows=(
                WindowCharge(
                    entity_type="key",
                    entity_id=api_key,
                    window_duration="1d",
                    window_start=old_start,
                    spend=0.3,
                    started_at=old_start,
                ),
            ),
        )
        current: Final = event(
            "window-current", request_id=current_request, spend=0.3,
            payload_json=first_effects.model_dump_json()
        )
        await _store(db).append(current)

        async def apply(tx: SupportsRawQueries, batch: tuple[AccountingEvent, ...]) -> None:
            await apply_accounting_events(tx, batch)

        concurrent: Final = await asyncio.gather(_store(db).flush(apply), _store(db).flush(apply))
        assert sum(concurrent) == 1
        window_rows: Final = await db.query_raw(
            'SELECT spend FROM "LiteLLM_BudgetWindowSpend" '
            'WHERE entity_type = $1 AND entity_id = $2 AND window_duration = $3',
            "key",
            api_key,
            "1d",
        )
        assert float(window_rows[0]["spend"]) == pytest.approx(0.5)

        newer_effects: Final = first_effects.model_copy(
            update={
                "windows": (
                    first_effects.windows[0].model_copy(update={"window_start": new_start, "spend": 0.4}),
                )
            }
        )
        await _store(db).append(
            event("window-new", request_id=f"window-new-{run}", spend=0.4,
                  payload_json=newer_effects.model_dump_json())
        )
        assert await _store(db).flush(apply) == 1
        older_effects: Final = first_effects.model_copy(
            update={"windows": (first_effects.windows[0].model_copy(update={"spend": 0.1}),)}
        )
        late_old: Final = event(
            "window-late-old",
            request_id=f"window-old-{run}",
            spend=0.1,
            payload_json=older_effects.model_dump_json(),
        )
        await _store(db).append(late_old)
        with pytest.raises(RuntimeError, match="pre-commit"):
            await _store(db, before_commit=True).flush(apply)
        rolled_back_window: Final = await db.query_raw(
            'SELECT spend, window_start::text AS window_start FROM "LiteLLM_BudgetWindowSpend" '
            'WHERE entity_type = $1 AND entity_id = $2 AND window_duration = $3',
            "key",
            api_key,
            "1d",
        )
        rolled_back_event: Final = await db.query_raw(
            'SELECT processed_at FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1', late_old.event_key
        )
        assert float(rolled_back_window[0]["spend"]) == pytest.approx(0.4)
        assert str(rolled_back_window[0]["window_start"]).startswith(new_start[:10])
        assert rolled_back_event[0]["processed_at"] is None
        assert await _store(db).flush(apply) == 1
        final_rows: Final = await db.query_raw(
            'SELECT spend, window_start::text AS window_start FROM "LiteLLM_BudgetWindowSpend" '
            'WHERE entity_type = $1 AND entity_id = $2 AND window_duration = $3',
            "key",
            api_key,
            "1d",
        )
        expected_final_spend: Final = 0.4 + older_effects.windows[0].spend
        assert float(final_rows[0]["spend"]) == pytest.approx(expected_final_spend)
        assert str(final_rows[0]["window_start"]).startswith(new_start[:10])
        processed: Final = await db.query_raw(
            'SELECT processed_at FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1', late_old.event_key
        )
        assert processed[0]["processed_at"] is not None
        assert await _store(db).flush(apply) == 0
        replayed_rows: Final = await db.query_raw(
            'SELECT spend FROM "LiteLLM_BudgetWindowSpend" '
            'WHERE entity_type = $1 AND entity_id = $2 AND window_duration = $3',
            "key",
            api_key,
            "1d",
        )
        assert float(replayed_rows[0]["spend"]) == pytest.approx(expected_final_spend)
    finally:
        await db.execute_raw('DELETE FROM "LiteLLM_BudgetWindowSpend" WHERE entity_id = $1', api_key)
        await db.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = $1', legacy_request)


async def test_writer_claim_commits_log_and_event_and_handles_batch_cost_replay(
    db: Prisma, event: Callable[..., AccountingEvent]
) -> None:
    run: Final = uuid.uuid4().hex
    request_id: Final = f"batch-claim-{run}"
    call_type: Final = CallTypes.aretrieve_batch.value
    start: Final = "2199-04-01T00:00:00+00:00"
    row: Final = {
        "request_id": request_id,
        "call_type": call_type,
        "api_key": f"batch-key-{run}",
        "spend": 0.0,
        "startTime": start,
        "endTime": start,
        "status": "success",
    }
    zero: Final = event("batch-zero", request_id=request_id, call_type=call_type, spend=0.0)
    paid_row: Final = {**row, "spend": 0.7}
    paid: Final = event("batch-paid", request_id=request_id, call_type=call_type, spend=0.7)
    duplicate: Final = event("batch-legacy-duplicate", request_id=request_id, call_type=call_type, spend=0.7)
    disabled_request: Final = f"batch-disabled-{run}"
    disabled: Final = event("batch-disabled", request_id=disabled_request, call_type=call_type, spend=0.2)
    no_log_request: Final = f"no-log-{run}"
    no_log_event: Final = event("no-log", request_id=no_log_request, spend=0.2)
    disabled_row: Final = {
        key: value
        for key, value in row.items()
        if key in ("request_id", "call_type", "spend", "startTime", "endTime", "status")
    } | {"request_id": disabled_request, "spend": 0.2}

    async def accept_zero(tx: SupportsRawQueries, candidate: AccountingEvent) -> bool:
        return await _accept_accounting_spend_log(tx, candidate, row, True)

    async def accept_paid(tx: SupportsRawQueries, candidate: AccountingEvent) -> bool:
        return await _accept_accounting_spend_log(tx, candidate, paid_row, True)

    async def accept_disabled(tx: SupportsRawQueries, candidate: AccountingEvent) -> bool:
        return await _accept_accounting_spend_log(
            tx,
            candidate,
            disabled_row,
            False,
        )

    async def accept_without_logs(tx: SupportsRawQueries, candidate: AccountingEvent) -> bool:
        return await _accept_accounting_spend_log(tx, candidate, {**row, "request_id": no_log_request}, False)

    try:
        with pytest.raises(RuntimeError, match="pre-commit"):
            await _store(db, before_commit=True).append(zero, accept=accept_zero)
        assert await db.query_raw('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = $1', request_id) == []
        assert await _event_count(db, (zero.event_key,)) == 0
        assert await _store(db).append(zero, accept=accept_zero) is True
        assert await _store(db).append(paid, accept=accept_paid) is True
        assert await _store(db).append(duplicate, accept=accept_paid) is False
        paid_logs: Final = await db.query_raw(
            'SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id = $1', request_id
        )
        assert float(paid_logs[0]["spend"]) == pytest.approx(0.7)
        assert await _store(db).append(disabled, accept=accept_disabled) is True
        assert await db.query_raw(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = $1', disabled_request
        ) != []
        disabled_logs: Final = await db.query_raw(
            'SELECT api_key, model FROM "LiteLLM_SpendLogs" WHERE request_id = $1', disabled_request
        )
        assert disabled_logs[0] == {"api_key": "", "model": ""}
        assert await _store(db).append(no_log_event, accept=accept_without_logs) is True
        assert await db.query_raw(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = $1', no_log_request
        ) == []
    finally:
        await db.execute_raw(
            'DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = ANY($1::text[])',
            [request_id, disabled_request, no_log_request],
        )


async def test_writer_admission_is_atomic_and_restarted_monitor_applies_once(
    db: Prisma, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from litellm.proxy import proxy_server
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    from litellm.proxy.utils import monitor_spend_logs_queue

    run: Final = uuid.uuid4().hex
    request_id: Final = f"writer-entry-{run}"
    user_id: Final = f"writer-user-{run}"
    tag: Final = f"writer-tag-{run}"
    event_key: Final = json.dumps((request_id, "success", "ordinary"))
    monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
    client: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(UserApiKeyCache()))
    writer: Final = DBSpendUpdateWriter()
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    monkeypatch.setattr(proxy_server, "litellm_proxy_budget_name", "")
    observed: Final = asyncio.get_running_loop().create_future()
    start: Final = datetime.fromisoformat("2199-05-06T12:00:00+00:00")

    async def inspect_uncommitted_admission() -> None:
        events: Final = await db.query_raw(
            'SELECT event_key FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1', event_key
        )
        logs: Final = await db.query_raw(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = $1', request_id
        )
        observed.set_result((len(events), len(logs)))
        raise RuntimeError("cache refresh failed after durable admission")

    try:
        await client.db.connect()
        await db.execute_raw(
            'INSERT INTO "LiteLLM_TagTable" (tag_name, models) VALUES ($1, ARRAY[]::text[])', tag
        )
        charged: Final = await writer.update_database(
            token=None,
            user_id=None,
            end_user_id=None,
            team_id=None,
            org_id=None,
            project_id=None,
            kwargs=_writer_kwargs(user_id, tag),
            completion_response={
                "id": request_id,
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
            start_time=start,
            end_time=start,
            response_cost=0.6,
            on_accounting_admitted=inspect_uncommitted_admission,
        )
        assert charged is True
        assert observed.result() == (0, 0)
        assert await db.query_raw(
            'SELECT 1 FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1', event_key
        ) != []
        assert await db.query_raw(
            'SELECT 1 FROM "LiteLLM_SpendLogs" WHERE request_id = $1', request_id
        ) != []
        assert client.spend_log_transactions == []
        assert client.model_usage_transactions == []
        assert client.tool_usage_transactions == []

        next_logging: Final = ProxyLogging(UserApiKeyCache())
        monitor: Final = asyncio.create_task(monitor_spend_logs_queue(client, None, next_logging))

        async def await_completion() -> None:
            while not await db.query_raw(
                'SELECT 1 FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1 AND processed_at IS NOT NULL',
                event_key,
            ):
                if monitor.done():
                    await monitor
                await asyncio.sleep(0)

        try:
            await asyncio.wait_for(await_completion(), timeout=5)
        finally:
            monitor.cancel()
            with pytest.raises(asyncio.CancelledError):
                await monitor

        assert await _entity_spend(db, "LiteLLM_TagTable", "tag_name", tag) == pytest.approx(0.6)
        daily_user: Final = await db.query_raw(
            'SELECT spend, api_requests FROM "LiteLLM_DailyUserSpend" WHERE user_id = $1', user_id
        )
        assert (float(daily_user[0]["spend"]), int(daily_user[0]["api_requests"])) == pytest.approx((0.6, 1))
        daily_global: Final = await db.query_raw(
            'SELECT spend, api_requests FROM "LiteLLM_DailyGlobalSpend" WHERE date = $1 AND model = $2',
            "2199-05-06",
            "gpt-4o-mini",
        )
        assert (float(daily_global[0]["spend"]), int(daily_global[0]["api_requests"])) == pytest.approx((0.6, 1))
        assert await DBSpendUpdateWriter().update_database(
            token=None,
            user_id=None,
            end_user_id=None,
            team_id=None,
            org_id=None,
            project_id=None,
            kwargs=_writer_kwargs(user_id, tag),
            completion_response={
                "id": request_id,
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
            start_time=start,
            end_time=start,
            response_cost=0.6,
        ) is False
        assert await DBSpendUpdateWriter().flush_accounting(client) == 0
        assert await _entity_spend(db, "LiteLLM_TagTable", "tag_name", tag) == pytest.approx(0.6)
    finally:
        await db.execute_raw('DELETE FROM "LiteLLM_DailyUserSpend" WHERE user_id = $1', user_id)
        await db.execute_raw('DELETE FROM "LiteLLM_DailyTagSpend" WHERE request_id = $1', request_id)
        await db.execute_raw(
            'DELETE FROM "LiteLLM_DailyGlobalSpend" WHERE date = $1 AND model = $2',
            "2199-05-06",
            "gpt-4o-mini",
        )
        await db.execute_raw('DELETE FROM "LiteLLM_TagTable" WHERE tag_name = $1', tag)
        await db.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = $1', request_id)
        await db.execute_raw('DELETE FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1', event_key)
        await client.db.disconnect()


async def test_cancelled_writer_admission_rolls_back_event_and_source_log(
    db: Prisma, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from litellm.proxy import proxy_server
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

    run: Final = uuid.uuid4().hex
    request_id: Final = f"writer-cancel-{run}"
    user_id: Final = f"writer-cancel-user-{run}"
    tag: Final = f"writer-cancel-tag-{run}"
    event_key: Final = json.dumps((request_id, "success", "ordinary"))
    monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
    client: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(UserApiKeyCache()))
    writer: Final = DBSpendUpdateWriter()
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    monkeypatch.setattr(proxy_server, "litellm_proxy_budget_name", "")
    entered: Final = asyncio.Event()
    observed: Final = asyncio.get_running_loop().create_future()
    start: Final = datetime.fromisoformat("2199-05-07T12:00:00+00:00")

    async def pause_after_uncommitted_admission() -> None:
        events: Final = await db.query_raw(
            'SELECT event_key FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1', event_key
        )
        logs: Final = await db.query_raw(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = $1', request_id
        )
        observed.set_result((len(events), len(logs)))
        entered.set()
        await asyncio.Event().wait()

    try:
        await client.db.connect()
        await db.execute_raw(
            'INSERT INTO "LiteLLM_TagTable" (tag_name, models) VALUES ($1, ARRAY[]::text[])', tag
        )
        task: Final = asyncio.create_task(
            writer.update_database(
                token=None,
                user_id=None,
                end_user_id=None,
                team_id=None,
                org_id=None,
                project_id=None,
                kwargs=_writer_kwargs(user_id, tag),
                completion_response={
                    "id": request_id,
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                },
                start_time=start,
                end_time=start,
                response_cost=0.6,
                on_accounting_admitted=pause_after_uncommitted_admission,
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert observed.result() == (0, 0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await db.query_raw(
            'SELECT 1 FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1', event_key
        ) == []
        assert await db.query_raw(
            'SELECT 1 FROM "LiteLLM_SpendLogs" WHERE request_id = $1', request_id
        ) == []
    finally:
        await db.execute_raw('DELETE FROM "LiteLLM_TagTable" WHERE tag_name = $1', tag)
        await db.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = $1', request_id)
        await db.execute_raw('DELETE FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1', event_key)
        await client.db.disconnect()


async def test_logging_waits_for_database_spend_while_external_success_callback_is_blocked(
    db: Prisma, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.hidden_params import set_hidden_param
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.proxy import proxy_server
    from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
    from litellm.types.utils import ModelResponse

    run: Final = uuid.uuid4().hex
    request_id: Final = f"logging-accounting-{run}"
    user_id: Final = f"logging-accounting-user-{run}"
    tag: Final = f"logging-accounting-tag-{run}"
    day: Final = "2199-05-08"
    event_key: Final = json.dumps((request_id, "success", "ordinary"))
    start: Final = datetime.fromisoformat(f"{day}T12:00:00+00:00")
    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()

    class BlockedExternalLogger(CustomLogger):
        @property
        def requires_success_await(self) -> bool:
            return False

        async def async_log_success_event(
            self, kwargs: object, response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            entered.set()
            await release.wait()

    monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
    client: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(UserApiKeyCache()))
    proxy_logger: Final = _ProxyDBLogger()
    external_logger: Final = BlockedExternalLogger()
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", client.proxy_logging_obj)
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    monkeypatch.setattr(proxy_server, "litellm_proxy_budget_name", "")
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    try:
        await client.db.connect()
        await db.execute_raw(
            'INSERT INTO "LiteLLM_TagTable" (tag_name, models) VALUES ($1, ARRAY[]::text[])', tag
        )
        params: Final = {
            "metadata": {
                "user_api_key_user_id": user_id,
                "tags": [tag],
                "status": "success",
            }
        }
        logging_obj: Final = Logging(
            model="contract-model",
            messages=[{"role": "user", "content": "contract"}],
            stream=False,
            call_type=CallTypes.acompletion.value,
            start_time=start,
            litellm_call_id=request_id,
            function_id="accounting-contract",
            dynamic_async_success_callbacks=[proxy_logger, external_logger],
            kwargs={"litellm_params": params},
        )
        logging_obj.update_environment_variables(
            litellm_params=params,
            optional_params={},
            model="contract-model",
            custom_llm_provider="openai",
        )
        response: Final = ModelResponse(
            id=request_id,
            model="contract-model",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )
        set_hidden_param(response, "response_cost", 0.73)

        assert logging_obj.requires_success_await is True
        returned: Final = await litellm.acompletion(
            model="openai/contract-model",
            messages=[{"role": "user", "content": "contract"}],
            mock_response=response,
            metadata=params["metadata"],
            litellm_logging_obj=logging_obj,
        )
        assert returned.id == request_id
        await asyncio.wait_for(entered.wait(), timeout=5)
        persisted_logs: Final = await db.query_raw(
            'SELECT request_id, spend FROM "LiteLLM_SpendLogs" WHERE request_id = $1', request_id
        )
        persisted_events: Final = await db.query_raw(
            'SELECT event_key, processed_at FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1', event_key
        )
        assert len(persisted_logs) == 1
        assert float(persisted_logs[0]["spend"]) == pytest.approx(0.73)
        assert len(persisted_events) == 1
        assert persisted_events[0]["processed_at"] is None
    finally:
        release.set()
        await GLOBAL_LOGGING_WORKER.flush()
        await db.execute_raw('DELETE FROM "LiteLLM_DailyUserSpend" WHERE user_id = $1', user_id)
        await db.execute_raw('DELETE FROM "LiteLLM_DailyTagSpend" WHERE request_id = $1', request_id)
        await db.execute_raw(
            'DELETE FROM "LiteLLM_DailyGlobalSpend" WHERE date = $1 AND model = $2', day, "contract-model"
        )
        await db.execute_raw('DELETE FROM "LiteLLM_TagTable" WHERE tag_name = $1', tag)
        await db.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = $1', request_id)
        await db.execute_raw('DELETE FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1', event_key)
        await client.db.disconnect()


async def test_database_accounting_failure_does_not_retry_provider_or_run_failure_callbacks(
    db: Prisma, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.hidden_params import set_hidden_param
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.proxy import proxy_server
    from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
    from litellm.types.utils import ModelResponse

    run: Final = uuid.uuid4().hex
    request_id: Final = f"sdk-accounting-failure-{run}"
    user_id: Final = f"sdk-accounting-failure-user-{run}"
    tag: Final = f"sdk-accounting-failure-tag-{run}"
    event_key: Final = json.dumps((request_id, "success", "ordinary"))
    trigger_name: Final = f"accounting_failure_{run}"
    function_name: Final = f"accounting_failure_fn_{run}"
    start: Final = datetime.fromisoformat("2199-05-09T12:00:00+00:00")
    failure_called: Final = asyncio.Event()

    class FailureObserver(CustomLogger):
        async def async_log_failure_event(
            self, kwargs: object, response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            failure_called.set()

    monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
    client: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(UserApiKeyCache()))
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", client.proxy_logging_obj)
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    monkeypatch.setattr(proxy_server, "litellm_proxy_budget_name", "")
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    try:
        await client.db.connect()
        await db.execute_raw(
            'INSERT INTO "LiteLLM_TagTable" (tag_name, models) VALUES ($1, ARRAY[]::text[])', tag
        )
        await db.execute_raw(
            f'''CREATE FUNCTION "{function_name}"() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN
                    IF NEW.request_id = '{request_id}' THEN
                        RAISE EXCEPTION 'injected accounting admission failure';
                    END IF;
                    RETURN NEW;
                END
            $$'''
        )
        await db.execute_raw(
            f'''CREATE TRIGGER "{trigger_name}" BEFORE INSERT ON "LiteLLM_SpendAccountingEvent"
                FOR EACH ROW EXECUTE FUNCTION "{function_name}"()'''
        )
        params: Final = {
            "metadata": {
                "user_api_key_user_id": user_id,
                "tags": [tag],
                "status": "success",
            }
        }
        logging_obj: Final = Logging(
            model="contract-model",
            messages=[{"role": "user", "content": "contract"}],
            stream=False,
            call_type=CallTypes.acompletion.value,
            start_time=start,
            litellm_call_id=request_id,
            function_id="accounting-contract-failure",
            dynamic_async_success_callbacks=[_ProxyDBLogger()],
            dynamic_async_failure_callbacks=[FailureObserver()],
            kwargs={"litellm_params": params},
        )
        logging_obj.update_environment_variables(
            litellm_params=params,
            optional_params={},
            model="contract-model",
            custom_llm_provider="openai",
        )
        response: Final = ModelResponse(
            id=request_id,
            model="contract-model",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )
        set_hidden_param(response, "response_cost", 0.73)

        with pytest.raises(Exception, match="injected accounting admission failure"):
            await litellm.acompletion(
                model="openai/contract-model",
                messages=[{"role": "user", "content": "contract"}],
                mock_response=response,
                metadata=params["metadata"],
                litellm_logging_obj=logging_obj,
                num_retries=2,
            )
        assert failure_called.is_set() is False
        assert await db.query_raw(
            'SELECT 1 FROM "LiteLLM_SpendLogs" WHERE request_id = $1', request_id
        ) == []
        assert await db.query_raw(
            'SELECT 1 FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1', event_key
        ) == []
    finally:
        await GLOBAL_LOGGING_WORKER.flush()
        await db.execute_raw(f'DROP TRIGGER IF EXISTS "{trigger_name}" ON "LiteLLM_SpendAccountingEvent"')
        await db.execute_raw(f'DROP FUNCTION IF EXISTS "{function_name}"()')
        await db.execute_raw('DELETE FROM "LiteLLM_DailyUserSpend" WHERE user_id = $1', user_id)
        await db.execute_raw('DELETE FROM "LiteLLM_DailyTagSpend" WHERE request_id = $1', request_id)
        await db.execute_raw(
            'DELETE FROM "LiteLLM_DailyGlobalSpend" WHERE date = $1 AND model = $2', "2199-05-09", "contract-model"
        )
        await db.execute_raw('DELETE FROM "LiteLLM_TagTable" WHERE tag_name = $1', tag)
        await db.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = $1', request_id)
        await db.execute_raw('DELETE FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1', event_key)
        await client.db.disconnect()


async def test_cancelled_sdk_waits_for_blocked_accounting_commit(
    db: Prisma, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.hidden_params import set_hidden_param
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.proxy import proxy_server
    from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
    from litellm.proxy.spend_tracking.spend_event import ObjectMapping
    from litellm.types.utils import ModelResponse

    run: Final = uuid.uuid4().hex
    request_id: Final = f"sdk-accounting-cancel-{run}"
    user_id: Final = f"sdk-accounting-cancel-user-{run}"
    tag: Final = f"sdk-accounting-cancel-tag-{run}"
    event_key: Final = json.dumps((request_id, "success", "ordinary"))
    start: Final = datetime.fromisoformat("2199-05-10T12:00:00+00:00")
    admission_started: Final = asyncio.Event()
    release_admission: Final = asyncio.Event()

    class PausedAccountingLogger(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self._database_logger: Final = _ProxyDBLogger()

        @property
        def requires_success_await(self) -> bool:
            return True

        async def async_log_success_event(
            self, kwargs: ObjectMapping, response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            admission_started.set()
            await release_admission.wait()
            await self._database_logger.async_log_success_event(
                kwargs=kwargs, response_obj=response_obj, start_time=start_time, end_time=end_time
            )

    monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
    client: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(UserApiKeyCache()))
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", client.proxy_logging_obj)
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    monkeypatch.setattr(proxy_server, "litellm_proxy_budget_name", "")
    monkeypatch.setattr(litellm, "_async_success_callback", [])

    try:
        await client.db.connect()
        await db.execute_raw(
            'INSERT INTO "LiteLLM_TagTable" (tag_name, models) VALUES ($1, ARRAY[]::text[])', tag
        )
        params: Final = {
            "metadata": {
                "user_api_key_user_id": user_id,
                "tags": [tag],
                "status": "success",
            }
        }
        logging_obj: Final = Logging(
            model="contract-model",
            messages=[{"role": "user", "content": "contract"}],
            stream=False,
            call_type=CallTypes.acompletion.value,
            start_time=start,
            litellm_call_id=request_id,
            function_id="accounting-contract-cancel",
            dynamic_async_success_callbacks=[PausedAccountingLogger()],
            kwargs={"litellm_params": params},
        )
        logging_obj.update_environment_variables(
            litellm_params=params,
            optional_params={},
            model="contract-model",
            custom_llm_provider="openai",
        )
        response: Final = ModelResponse(
            id=request_id,
            model="contract-model",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )
        set_hidden_param(response, "response_cost", 0.73)

        assert logging_obj.requires_success_await is True
        sdk_call: Final = asyncio.create_task(
            litellm.acompletion(
                model="openai/contract-model",
                messages=[{"role": "user", "content": "contract"}],
                mock_response=response,
                metadata=params["metadata"],
                litellm_logging_obj=logging_obj,
            )
        )
        try:
            await asyncio.wait_for(admission_started.wait(), timeout=5)
        except TimeoutError:
            if sdk_call.done():
                await sdk_call
            raise
        sdk_call.cancel()
        await asyncio.sleep(0.05)
        assert sdk_call.done() is False
        release_admission.set()

        with pytest.raises(asyncio.CancelledError):
            await sdk_call
        persisted_logs: Final = await db.query_raw(
            'SELECT request_id, spend FROM "LiteLLM_SpendLogs" WHERE request_id = $1', request_id
        )
        persisted_events: Final = await db.query_raw(
            'SELECT event_key FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1', event_key
        )
        assert len(persisted_logs) == 1
        assert float(persisted_logs[0]["spend"]) == pytest.approx(0.73)
        assert len(persisted_events) == 1
    finally:
        await GLOBAL_LOGGING_WORKER.flush()
        await db.execute_raw('DELETE FROM "LiteLLM_DailyUserSpend" WHERE user_id = $1', user_id)
        await db.execute_raw('DELETE FROM "LiteLLM_DailyTagSpend" WHERE request_id = $1', request_id)
        await db.execute_raw(
            'DELETE FROM "LiteLLM_DailyGlobalSpend" WHERE date = $1 AND model = $2', "2199-05-10", "contract-model"
        )
        await db.execute_raw('DELETE FROM "LiteLLM_TagTable" WHERE tag_name = $1', tag)
        await db.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = $1', request_id)
        await db.execute_raw('DELETE FROM "LiteLLM_SpendAccountingEvent" WHERE event_key = $1', event_key)
        await client.db.disconnect()

