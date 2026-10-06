"""Tests for the single-statement daily spend upsert (LIT-5291)."""

import re
import sqlite3
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Final, cast
from unittest.mock import MagicMock

import pytest

from litellm.proxy._types import DailyUserSpendTransaction
from litellm.proxy.db.baseline_accounting import DailyBaselineAttribution, DailyBaselineTarget
from litellm.proxy.db.daily_spend_bulk_upsert import (
    DAILY_SPEND_TABLES,
    SqlValue,
    build_bulk_upsert,
    conflict_key,
    merge_by_conflict_key,
)
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.db.rollup_lock_timeout import ROLLUP_LOCK_TIMEOUT_SQL
from litellm.proxy.utils import PrismaClient, ProxyLogging

TAG_TABLE = DAILY_SPEND_TABLES["tag"]
USER_TABLE = DAILY_SPEND_TABLES["user"]

# Every nullable member of the unique constraint, so a test that only varied the provider
# cannot pass while a sibling column still leaks a NULL into the conflict target.
NULLABLE_KEY_COLUMNS = ("model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint")


def tag_txn(**overrides):
    return {
        "tag": "team-a",
        "date": "2026-08-10",
        "api_key": "sk-hash",
        "model": "gpt-4o-mini",
        "model_group": "gpt-4o-mini",
        "custom_llm_provider": "openai",
        "mcp_namespaced_tool_name": "",
        "endpoint": "/chat/completions",
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "spend": 0.25,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
        "request_id": "req-1",
        **overrides,
    }


@pytest.mark.parametrize("column", NULLABLE_KEY_COLUMNS)
def test_conflict_key_normalizes_every_nullable_key_column(column):
    """A NULL member can never match itself in a unique index, so the row would be
    re-inserted on every flush. Each nullable key column must arrive as ''."""
    key = conflict_key(TAG_TABLE, tag_txn(**{column: None}))

    assert "" in key
    assert None not in key
    assert key == conflict_key(TAG_TABLE, tag_txn(**{column: ""}))


@pytest.mark.parametrize("order", [("null_first"), ("empty_first")])
def test_null_and_empty_provider_merge_into_one_row(order):
    """Two queue entries differing only in NULL versus '' arbitrate to the same row.
    Postgres rejects one statement touching a row twice, so they must be folded first.
    Asserted under both input orders: a single ordering would prove nothing here."""
    null_entry = tag_txn(custom_llm_provider=None, spend=0.25, api_requests=1)
    empty_entry = tag_txn(custom_llm_provider="", spend=0.75, api_requests=3)
    transactions = (null_entry, empty_entry) if order == "null_first" else (empty_entry, null_entry)

    merged = merge_by_conflict_key(TAG_TABLE, transactions)

    assert len(merged) == 1
    _, folded = merged[0]
    assert folded["spend"] == pytest.approx(1.0)
    assert folded["api_requests"] == 4


def test_distinct_keys_are_not_merged_and_are_ordered_deterministically():
    unordered = (tag_txn(tag="z-team"), tag_txn(tag="a-team"), tag_txn(tag="m-team"))

    merged = merge_by_conflict_key(TAG_TABLE, unordered)

    assert [txn["tag"] for _, txn in merged] == ["a-team", "m-team", "z-team"]
    assert merged == merge_by_conflict_key(TAG_TABLE, tuple(reversed(unordered)))


def test_one_statement_carries_every_row_in_the_batch():
    batch = merge_by_conflict_key(TAG_TABLE, tuple(tag_txn(tag=f"team-{i}") for i in range(100)))

    sql, params = build_bulk_upsert(TAG_TABLE, batch)

    assert sql.count("INSERT INTO") == 1
    assert len(re.findall(r"ON CONFLICT", sql)) == 1
    # 25 bound columns per row plus the inlined updated_at, so the row count is what
    # separates one multi-row statement from a hundred single-row ones.
    assert len(params) == 100 * 25
    assert "$2500::text" in sql
    assert sql.count("(NOW() AT TIME ZONE 'UTC')") == 100 + 1


def test_conflict_target_is_the_full_unique_constraint():
    sql, _ = build_bulk_upsert(TAG_TABLE, merge_by_conflict_key(TAG_TABLE, (tag_txn(),)))

    conflict_target = re.search(r"ON CONFLICT \(([^)]*)\)", sql)
    assert conflict_target is not None
    assert conflict_target.group(1) == (
        '"tag", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint"'
    )


@pytest.mark.parametrize(
    "column",
    [
        "prompt_tokens",
        "completion_tokens",
        "spend",
        "api_requests",
        "successful_requests",
        "failed_requests",
        "total_response_time_ms",
        "timed_requests",
    ],
)
def test_counters_increment_rather_than_overwrite(column):
    """An overwrite would silently discard every earlier flush's spend for that row."""
    sql, _ = build_bulk_upsert(TAG_TABLE, merge_by_conflict_key(TAG_TABLE, (tag_txn(),)))

    assert f'"{column}" = "LiteLLM_DailyTagSpend"."{column}" + EXCLUDED."{column}"' in sql


def test_request_id_is_preserved_when_a_later_batch_carries_none():
    sql, params = build_bulk_upsert(TAG_TABLE, merge_by_conflict_key(TAG_TABLE, (tag_txn(request_id=None),)))

    assert '"request_id" = COALESCE(EXCLUDED."request_id", "LiteLLM_DailyTagSpend"."request_id")' in sql
    assert None in params


def test_non_tag_tables_carry_no_request_id_column():
    user_txn = {**tag_txn(), "user_id": "u-1"}
    del user_txn["tag"]

    sql, _ = build_bulk_upsert(USER_TABLE, merge_by_conflict_key(USER_TABLE, (user_txn,)))

    assert "request_id" not in sql
    assert '"user_id"' in sql


class _RecordingDb:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[object, ...]]] = []
        self.session_settings: list[tuple[str, tuple[object, ...]]] = []

    async def execute_raw(self, query: str, *args: object) -> int:
        if query == ROLLUP_LOCK_TIMEOUT_SQL:
            self.session_settings.append((query, args))
            return 0
        self.statements.append((query, args))
        return len(args)

    @asynccontextmanager
    async def _tx(self) -> AsyncIterator["_RecordingDb"]:
        yield self

    def tx(self, timeout: object = None) -> AbstractAsyncContextManager["_RecordingDb"]:
        return self._tx()


class _RecordingPrismaClient:
    def __init__(self) -> None:
        self.db = _RecordingDb()


@pytest.mark.asyncio
async def test_writer_issues_one_statement_per_batch_not_one_per_key():
    """The whole point of LIT-5291: 250 aggregated keys must not become 250 statements."""
    prisma_client = _RecordingPrismaClient()
    transactions = {f"k{i}": tag_txn(tag=f"team-{i}") for i in range(250)}

    await DBSpendUpdateWriter.update_daily_tag_spend(
        n_retry_times=0,
        prisma_client=prisma_client,
        proxy_logging_obj=None,
        daily_spend_transactions=transactions,
    )

    # 250 keys at a batch size of 100 is three statements, one per batch.
    assert len(prisma_client.db.statements) == 3
    assert [statement.count("ON CONFLICT") for statement, _ in prisma_client.db.statements] == [1, 1, 1]
    assert transactions == {}


@pytest.mark.asyncio
async def test_writer_survives_a_transaction_whose_key_columns_are_null():
    """A NULL key column used to raise out of prisma and drop the whole batch's spend."""
    prisma_client = _RecordingPrismaClient()
    transactions = {
        "mcp": tag_txn(model=None, custom_llm_provider=None, mcp_namespaced_tool_name="server/tool"),
        "chat": tag_txn(),
    }

    await DBSpendUpdateWriter.update_daily_tag_spend(
        n_retry_times=0,
        prisma_client=prisma_client,
        proxy_logging_obj=None,
        daily_spend_transactions=transactions,
    )

    assert len(prisma_client.db.statements) == 1
    _, params = prisma_client.db.statements[0]
    assert None not in params[:9]
    assert transactions == {}


_ROLLUP_METRICS: Final = {
    "prompt_tokens": 11,
    "completion_tokens": 7,
    "api_requests": 3,
    "successful_requests": 2,
    "failed_requests": 1,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 4,
    "compression_saved_tokens": 9,
    "total_response_time_ms": 120,
    "timed_requests": 2,
    "spend": 0.25,
    "compression_savings_spend": 0.5,
    "prompt_caching_savings_spend": 0.75,
    "gateway_injected_caching_savings_spend": 1.25,
    "autorouter_savings_spend": 1.5,
}
_USER_KEYS: Final = (
    '"user_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint"'
)
_GLOBAL_KEYS: Final = '"date", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint"'


def _rollup_row(user_id: str) -> dict[str, SqlValue]:
    return {
        "user_id": user_id,
        "date": "2001-02-03",
        "api_key": "fixture-key",
        "model": "fixture-model",
        "model_group": "fixture-group",
        "custom_llm_provider": "fixture-provider",
        "mcp_namespaced_tool_name": "fixture/tool",
        "endpoint": "/fixture",
        **_ROLLUP_METRICS,
    }


def _sqlite_sql(sql: str) -> str:
    return re.sub(r"::(?:double precision|bigint|text)", "", sql).replace(
        "(NOW() AT TIME ZONE 'UTC')", "'2001-02-03 04:05:06'"
    )


class _SqliteRollupDb:
    """Execute PostgreSQL's data-modifying CTEs as statements in one SQLite transaction."""

    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        metric_columns: Final = ", ".join(f'"{column}" NUMERIC NOT NULL' for column in _ROLLUP_METRICS)
        for table, keys, entity_columns in (
            ("LiteLLM_DailyUserSpend", _USER_KEYS, '"user_id" TEXT, "api_key" TEXT,'),
            ("LiteLLM_DailyGlobalSpend", _GLOBAL_KEYS, ""),
        ):
            self.connection.execute(
                f'CREATE TABLE "{table}" ("id" TEXT PRIMARY KEY, {entity_columns}'
                '"date" TEXT, "model" TEXT, "model_group" TEXT, "custom_llm_provider" TEXT,'
                '"mcp_namespaced_tool_name" TEXT, "endpoint" TEXT,'
                f'{metric_columns}, "updated_at" TEXT, UNIQUE ({keys}))'
            )

    async def execute_raw(self, query: str, *args: SqlValue) -> int:
        if query == ROLLUP_LOCK_TIMEOUT_SQL:
            return 0
        input_sql, remaining_sql = _sqlite_sql(query).split("),\nupdated_daily_users AS (\n", 1)
        input_columns, input_values = input_sql.removeprefix("WITH daily_user_input (").split(") AS (VALUES ", 1)
        self.connection.execute(f"CREATE TEMP TABLE daily_user_input ({input_columns})")
        self.connection.execute(
            f"INSERT INTO daily_user_input VALUES {input_values}",
            {str(index): value for index, value in enumerate(args, 1)},
        )
        user_sql, remaining_global_sql = remaining_sql.split("\nRETURNING ", 1)
        returned_columns, global_sql = remaining_global_sql.split("\n)\n", 1)
        returned_rows: Final = self.connection.execute(user_sql + " RETURNING " + returned_columns).fetchall()
        self.connection.execute(f"CREATE TEMP TABLE updated_daily_users ({returned_columns})")
        placeholders: Final = ", ".join("?" for _ in returned_columns.split(", "))
        self.connection.executemany(f"INSERT INTO updated_daily_users VALUES ({placeholders})", returned_rows)
        self.connection.execute(global_sql)
        self.connection.execute("DROP TABLE updated_daily_users")
        self.connection.execute("DROP TABLE daily_user_input")
        return len(returned_rows)

    @asynccontextmanager
    async def _tx(self) -> AsyncIterator["_SqliteRollupDb"]:
        self.connection.execute("BEGIN")
        try:
            yield self
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def tx(self, timeout: object = None) -> AbstractAsyncContextManager["_SqliteRollupDb"]:
        return self._tx()


class _SqliteRollupClient:
    def __init__(self) -> None:
        self.db = _SqliteRollupDb()


async def _flush_rollup(client: _SqliteRollupClient, rows: tuple[dict[str, SqlValue], ...]) -> None:
    transactions: Final = {str(index): row for index, row in enumerate(rows)}
    await DBSpendUpdateWriter.update_daily_user_spend(
        n_retry_times=0,
        prisma_client=cast(PrismaClient, client),
        proxy_logging_obj=cast(ProxyLogging, MagicMock(spec=ProxyLogging)),
        daily_spend_transactions=cast(dict[str, DailyUserSpendTransaction], transactions),
    )
    assert transactions == {}


@pytest.mark.asyncio
async def test_user_flush_keeps_global_metrics_equal_to_stored_user_groups_across_batches():
    client: Final = _SqliteRollupClient()
    await _flush_rollup(client, (_rollup_row("user-a"),))
    await _flush_rollup(
        client,
        (
            {**_rollup_row("user-a"), "model_group": "changed-label"},
            _rollup_row("user-b"),
            {**_rollup_row("user-b"), "api_key": "other-key"},
            {**_rollup_row("user-c"), "model_group": None, "endpoint": None},
            {**_rollup_row("user-c"), "model_group": "discarded-label", "endpoint": ""},
            {**_rollup_row("user-d"), "model_group": "", "endpoint": ""},
            {**_rollup_row("user-e"), "date": "2001-02-04"},
            {**_rollup_row("user-f"), "model": "other-model"},
            {**_rollup_row("user-g"), "custom_llm_provider": None},
            {**_rollup_row("user-h"), "mcp_namespaced_tool_name": None},
        ),
    )
    global_rows: Final = client.db.connection.execute(
        f'SELECT {_GLOBAL_KEYS}, {", ".join(_ROLLUP_METRICS)} FROM "LiteLLM_DailyGlobalSpend" ORDER BY {_GLOBAL_KEYS}'
    ).fetchall()
    user_groups: Final = ", ".join(
        f"COALESCE(\"{column}\", '')"
        for column in ("date", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint")
    )
    expected_rows: Final = client.db.connection.execute(
        f"SELECT {user_groups}, {', '.join(f'SUM({column})' for column in _ROLLUP_METRICS)} "
        f'FROM "LiteLLM_DailyUserSpend" GROUP BY {user_groups} ORDER BY {user_groups}'
    ).fetchall()
    assert tuple(map(tuple, global_rows)) == tuple(map(tuple, expected_rows))
    assert len(global_rows) == 6
    assert sum(row["spend"] for row in global_rows) == 11 * _ROLLUP_METRICS["spend"]
    assert {row["model_group"] for row in global_rows} == {"fixture-group", ""}
    assert (
        next(row for row in global_rows if row["endpoint"] == "")["api_requests"] == 3 * _ROLLUP_METRICS["api_requests"]
    )


@pytest.mark.asyncio
async def test_global_write_failure_rolls_back_the_user_increment():
    client: Final = _SqliteRollupClient()
    await _flush_rollup(client, (_rollup_row("user-a"),))
    client.db.connection.execute(
        'CREATE TRIGGER reject_global_write BEFORE INSERT ON "LiteLLM_DailyGlobalSpend" '
        "BEGIN SELECT RAISE(ABORT, 'controlled global failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="controlled global failure"):
        await _flush_rollup(client, (_rollup_row("user-a"), _rollup_row("user-b")))
    for table in ("LiteLLM_DailyUserSpend", "LiteLLM_DailyGlobalSpend"):
        (row,) = client.db.connection.execute(f'SELECT * FROM "{table}"').fetchall()
        for column, expected in _ROLLUP_METRICS.items():
            assert row[column] == expected


@pytest.mark.asyncio
async def test_baseline_savings_adjustments_update_global_without_changing_other_user_metrics():
    client: Final = _SqliteRollupClient()
    await _flush_rollup(client, (_rollup_row("user-a"),))
    attribution: Final = DailyBaselineAttribution(
        date="2001-02-03",
        api_key="fixture-key",
        model="fixture-model",
        model_group="changed-label",
        custom_llm_provider="fixture-provider",
        mcp_namespaced_tool_name="fixture/tool",
        endpoint="/fixture",
    )
    target: Final = DailyBaselineTarget(entity="user", entity_id="user-a")
    corrections: Final = (0.75, -1.25)
    for correction in corrections:
        adjustment: Final = attribution.adjustment(target, correction, "fixture-request")
        sql, params = build_bulk_upsert(USER_TABLE, merge_by_conflict_key(USER_TABLE, (adjustment,)))
        async with client.db.tx() as transaction:
            await transaction.execute_raw(sql, *params)
    for table in ("LiteLLM_DailyUserSpend", "LiteLLM_DailyGlobalSpend"):
        (row,) = client.db.connection.execute(f'SELECT * FROM "{table}"').fetchall()
        assert row["model_group"] == "fixture-group"
        for column, original in _ROLLUP_METRICS.items():
            expected: Final = original + sum(corrections) if column == "autorouter_savings_spend" else original
            assert row[column] == expected
