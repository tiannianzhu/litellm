import asyncio
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final, LiteralString
from uuid import uuid4
from zoneinfo import ZoneInfo

import psycopg
import pytest
import pytest_asyncio
from psycopg import sql
from psycopg.conninfo import make_conninfo

import litellm
from callbacks.search_quota import DAILY_LIMIT, DailySearchQuota, SearchQuota
from litellm.exceptions import RateLimitError
from litellm.types.utils import CallTypes


@dataclass(frozen=True)
class PostgresDatabase:
    dsn: str

    async def execute_raw(self, query: LiteralString, *args: str | int) -> int:
        parameters = {f"p{i}": value for i, value in enumerate(args, 1)}
        statement = re.sub(r"\$(\d+)", lambda match: f"%(p{match[1]})s", query)
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            cursor = await connection.execute(statement, parameters)
            return cursor.rowcount


@pytest_asyncio.fixture
async def database() -> AsyncIterator[PostgresDatabase]:
    dsn = os.environ.get("SEARCH_QUOTA_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("SEARCH_QUOTA_TEST_DATABASE_URL is required for PostgreSQL quota tests")
    schema = "search_quota_test_" + uuid4().hex
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            scoped_dsn = make_conninfo(dsn, options=f"-csearch_path={schema}")
            migration = (
                Path(__file__).resolve().parents[2]
                / "litellm-proxy-extras/litellm_proxy_extras/migrations"
                / "20260907020000_add_search_quota/migration.sql"
            )
            async with await psycopg.AsyncConnection.connect(scoped_dsn) as connection:
                await connection.execute(migration.read_text())
            yield PostgresDatabase(scoped_dsn)
        finally:
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.asyncio
async def test_quota_survives_recreation_and_resets_for_each_day_and_user(database: PostgresDatabase) -> None:
    quota: Final = DailySearchQuota(database)
    day: Final = date(2026, 9, 6)
    assert await quota.reserve("user:a", day, DAILY_LIMIT - 1)
    assert await DailySearchQuota(PostgresDatabase(database.dsn)).reserve("user:a", day, 1)
    assert not await DailySearchQuota(PostgresDatabase(database.dsn)).reserve("user:a", day, 1)
    assert await quota.reserve("user:b", day, 1)
    assert await quota.reserve("user:a", date(2026, 9, 7), 1)


@pytest.mark.asyncio
async def test_parallel_reservations_cannot_exceed_quota(database: PostgresDatabase) -> None:
    quota: Final = DailySearchQuota(database)
    day: Final = date(2026, 9, 6)
    assert await quota.reserve("user:a", day, DAILY_LIMIT - 2)
    results: Final = await asyncio.gather(*(quota.reserve("user:a", day, 1) for _ in range(20)))
    assert sum(results) == 2


@pytest.mark.asyncio
async def test_rejected_batch_does_not_consume_remaining_quota(database: PostgresDatabase) -> None:
    quota: Final = DailySearchQuota(database)
    day: Final = date(2026, 9, 6)
    assert not await quota.reserve("user:a", day, DAILY_LIMIT + 1)
    assert await quota.reserve("user:a", day, DAILY_LIMIT - 1)
    assert not await quota.reserve("user:a", day, 2)
    assert await quota.reserve("user:a", day, 1)


@pytest.mark.asyncio
async def test_hook_shares_quota_across_providers_and_sessions_but_allows_chat(
    database: PostgresDatabase,
) -> None:
    quota: Final = DailySearchQuota(database)
    callback: Final = SearchQuota(quota)
    day: Final = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    assert await quota.reserve("user:a", day, DAILY_LIMIT - 1)
    kwargs: Final[dict[str, object]] = {
        "search_provider": "bocha",
        "query": "first query",
        "session_id": "session-one",
        "litellm_metadata": {"user_api_key_end_user_id": "a", "user_api_key": "shared-key"},
    }
    await callback.async_pre_call_deployment_hook(kwargs, CallTypes.asearch)
    with pytest.raises(RateLimitError, match="daily quota"):
        await callback.async_pre_call_deployment_hook(
            {**kwargs, "query": "different query", "session_id": "session-two"}, CallTypes.asearch
        )
    await callback.async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)
    with pytest.raises(RateLimitError, match="daily quota"):
        await callback.async_pre_call_deployment_hook({**kwargs, "search_provider": "tavily"}, CallTypes.asearch)
    await callback.async_pre_call_deployment_hook(
        {**kwargs, "litellm_metadata": {"user_api_key_end_user_id": "b", "user_api_key": "shared-key"}},
        CallTypes.asearch,
    )


@pytest.mark.asyncio
async def test_unidentified_search_is_rejected_and_key_only_search_is_limited(database: PostgresDatabase) -> None:
    callback: Final = SearchQuota(DailySearchQuota(database))
    kwargs: Final[dict[str, object]] = {"search_provider": "bocha", "query": "query"}
    with pytest.raises(RateLimitError, match="identified"):
        await callback.async_pre_call_deployment_hook(kwargs, CallTypes.asearch)
    await callback.async_pre_call_deployment_hook(
        {**kwargs, "query": ["query"] * DAILY_LIMIT, "metadata": {"user_api_key": "shared-key"}}, CallTypes.asearch
    )
    with pytest.raises(RateLimitError, match="daily quota"):
        await callback.async_pre_call_deployment_hook(
            {**kwargs, "metadata": {"user_api_key": "shared-key"}}, CallTypes.asearch
        )


@pytest.mark.asyncio
async def test_asearch_stops_before_provider_call_when_quota_is_exhausted(database: PostgresDatabase) -> None:
    quota: Final = DailySearchQuota(database)
    callback: Final = SearchQuota(quota)
    day: Final = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    assert await quota.reserve("user:a", day, DAILY_LIMIT)
    original_callbacks: Final = litellm.callbacks
    litellm.callbacks = [callback]
    try:
        with pytest.raises(RateLimitError, match="daily quota"):
            await asyncio.wait_for(
                litellm.asearch(
                    query="quota enforcement probe",
                    search_provider="bocha",
                    api_key="unused",
                    api_base="http://127.0.0.1:1",
                    litellm_metadata={"user_api_key_end_user_id": "a"},
                    num_retries=0,
                ),
                timeout=5,
            )
    finally:
        litellm.callbacks = original_callbacks


class UnavailableDatabase:
    async def execute_raw(self, query: LiteralString, *args: str | int) -> int:
        raise ConnectionError("database unavailable")


@pytest.mark.asyncio
async def test_database_outage_blocks_search_but_not_chat() -> None:
    callback: Final = SearchQuota(DailySearchQuota(UnavailableDatabase()))
    kwargs: Final[dict[str, object]] = {
        "search_provider": "bocha",
        "query": "query",
        "litellm_metadata": {"user_api_key_end_user_id": "a"},
    }
    with pytest.raises(ConnectionError, match="database unavailable"):
        await callback.async_pre_call_deployment_hook(kwargs, CallTypes.asearch)
    await callback.async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)
