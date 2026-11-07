import asyncio
import re
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Final, LiteralString
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import httpx
import psycopg
import pytest
import pytest_asyncio
from fastapi import Request

import litellm
from litellm.deploy.compose.callbacks.search_quota import DAILY_LIMIT, DailySearchQuota, SearchQuota
from litellm.exceptions import RateLimitError
from litellm.llms.custom_httpx import llm_http_handler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.proxy.proxy_server import ProxyConfig
from litellm.types.utils import CallTypes
from tests.integration._support.database import PostgresTestDatabase, isolated_schema


@dataclass(frozen=True)
class PostgresDatabase:
    dsn: str = field(repr=False)

    async def execute_raw(self, query: LiteralString, *args: str | int) -> int:
        parameters = {f"p{i}": value for i, value in enumerate(args, 1)}
        statement = re.sub(r"\$(\d+)", lambda match: f"%(p{match[1]})s", query)
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            cursor = await connection.execute(statement, parameters)
            return cursor.rowcount


@pytest.fixture
def postgresql_database() -> Iterator[PostgresTestDatabase]:
    with isolated_schema() as database:
        yield database


@pytest_asyncio.fixture
async def database(postgresql_database: PostgresTestDatabase) -> AsyncIterator[PostgresDatabase]:
    migration: Final = (
        Path(__file__).resolve().parents[3]
        / "litellm/proxy-extras/litellm_proxy_extras/migrations"
        / "20260907020000_add_search_quota/migration.sql"
    )
    async with await psycopg.AsyncConnection.connect(postgresql_database.dsn) as connection:
        await connection.execute(migration.read_text())
    yield PostgresDatabase(postgresql_database.dsn)


@pytest.fixture
def registered_quota(database: PostgresDatabase) -> Iterator[DailySearchQuota]:
    quota: Final = DailySearchQuota(database)
    callback: Final = SearchQuota(quota)
    litellm.logging_callback_manager.add_litellm_callback(callback)
    try:
        yield quota
    finally:
        litellm.logging_callback_manager.remove_callback_from_all_lists(callback, require_self=False)


@pytest.fixture(autouse=True)
def clock() -> Iterator[Mock]:
    value: Final = Mock(
        wraps=datetime, **{"now.return_value": datetime(2026, 9, 6, 12, tzinfo=ZoneInfo("Asia/Shanghai"))}
    )
    with patch("litellm.deploy.compose.callbacks.search_quota.datetime", value):
        yield value


@pytest_asyncio.fixture
async def search_router(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[litellm.Router, Mock]]:
    provider: Final = Mock(return_value=httpx.Response(200, json={"results": []}))
    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(provider))
    monkeypatch.setattr(llm_http_handler, "get_async_httpx_client", lambda **_: handler)
    router: Final = litellm.Router(
        model_list=[],
        num_retries=0,
        search_tools=[
            {
                "search_tool_name": "test-search",
                "litellm_params": {
                    "search_provider": "deepseek",
                    "api_key": "test-key",
                    "api_base": "https://search.example/v1",
                },
            }
        ],
    )
    try:
        yield router, provider
    finally:
        router.retire()
        await handler.client.aclose()


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
    day: Final = date(2026, 9, 6)
    assert await quota.reserve("user:a", day, DAILY_LIMIT - 1)
    kwargs: Final[dict[str, object]] = {
        "search_provider": "deepseek",
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
    kwargs: Final[dict[str, object]] = {"search_provider": "deepseek", "query": "query"}
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
async def test_asearch_stops_before_provider_call_when_quota_is_exhausted(
    registered_quota: DailySearchQuota, search_router: tuple[litellm.Router, Mock]
) -> None:
    router, provider = search_router
    assert await registered_quota.reserve("user:a", date(2026, 9, 6), DAILY_LIMIT)
    response: Final = await asyncio.wait_for(
        router.asearch(
            search_tool_name="test-search",
            query="allowed query",
            litellm_metadata={"user_api_key_end_user_id": "b"},
        ),
        timeout=5,
    )
    assert response.object == "search"
    assert provider.call_count == 1

    with pytest.raises(RateLimitError, match="daily quota"):
        await asyncio.wait_for(
            litellm.asearch(
                query="quota enforcement probe",
                search_provider="deepseek",
                api_key="test-key",
                api_base="https://search.example/v1",
                litellm_metadata={"user_api_key_end_user_id": "a"},
                num_retries=0,
            ),
            timeout=5,
        )
    with pytest.raises(RateLimitError, match="daily quota"):
        await asyncio.wait_for(
            router.asearch(
                search_tool_name="test-search",
                query="quota enforcement probe",
                litellm_metadata={"user_api_key_end_user_id": "a"},
            ),
            timeout=5,
        )
    assert provider.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("nested_field", ["metadata", "litellm_metadata"])
async def test_authenticated_key_cannot_escape_quota_with_nested_end_user(
    registered_quota: DailySearchQuota,
    search_router: tuple[litellm.Router, Mock],
    postgresql_database: PostgresTestDatabase,
    nested_field: str,
) -> None:
    router, provider = search_router
    key: Final = "authenticated-key"
    identity: Final = f"key:{sha256(key.encode()).hexdigest()}"
    assert await registered_quota.reserve(identity, date(2026, 9, 6), DAILY_LIMIT)
    cleaned: Final[Mapping[str, object]] = await add_litellm_data_to_request(
        data={
            "query": "quota enforcement probe",
            "metadata": {"user_api_key_end_user_id": "forged-top-level"},
            "litellm_metadata": {"user_api_key_end_user_id": "forged-top-level"},
            "litellm_params": {nested_field: {"user_api_key_end_user_id": "forged-nested"}},
        },
        request=Request({"type": "http", "method": "POST", "path": "/v1/search", "headers": []}),
        user_api_key_dict=UserAPIKeyAuth(api_key=key),
        proxy_config=ProxyConfig(),
    )
    with pytest.raises(RateLimitError, match="daily quota"):
        await asyncio.wait_for(router.asearch(search_tool_name="test-search", **cleaned), timeout=5)
    assert provider.call_count == 0
    assert postgresql_database.connection.execute(
        'SELECT identity, calls FROM "LiteLLM_SearchQuota" ORDER BY identity'
    ).fetchall() == [(identity, DAILY_LIMIT)]


@pytest.mark.asyncio
async def test_search_quota_resets_only_at_beijing_midnight(
    registered_quota: DailySearchQuota,
    search_router: tuple[litellm.Router, Mock],
    postgresql_database: PostgresTestDatabase,
    clock: Mock,
) -> None:
    router, provider = search_router
    assert await registered_quota.reserve("user:a", date(2026, 9, 6), DAILY_LIMIT)
    clock.configure_mock(
        **{"now.side_effect": lambda tz: datetime(2026, 9, 6, 15, 59, 59, tzinfo=timezone.utc).astimezone(tz)}
    )
    with pytest.raises(RateLimitError, match="daily quota"):
        await router.asearch(
            search_tool_name="test-search", query="before midnight", litellm_metadata={"user_api_key_end_user_id": "a"}
        )
    assert provider.call_count == 0
    clock.configure_mock(**{"now.side_effect": lambda tz: datetime(2026, 9, 6, 16, tzinfo=timezone.utc).astimezone(tz)})
    response: Final = await router.asearch(
        search_tool_name="test-search", query="after midnight", litellm_metadata={"user_api_key_end_user_id": "a"}
    )
    assert response.object == "search"
    assert provider.call_count == 1
    assert postgresql_database.connection.execute(
        'SELECT day, identity, calls FROM "LiteLLM_SearchQuota" ORDER BY day'
    ).fetchall() == [(date(2026, 9, 6), "user:a", DAILY_LIMIT), (date(2026, 9, 7), "user:a", 1)]


class UnavailableDatabase:
    async def execute_raw(self, query: LiteralString, *args: str | int) -> int:
        raise ConnectionError("database unavailable")


@pytest.mark.asyncio
async def test_database_outage_blocks_search_but_not_chat() -> None:
    callback: Final = SearchQuota(DailySearchQuota(UnavailableDatabase()))
    kwargs: Final[dict[str, object]] = {
        "search_provider": "deepseek",
        "query": "query",
        "litellm_metadata": {"user_api_key_end_user_id": "a"},
    }
    with pytest.raises(ConnectionError, match="database unavailable"):
        await callback.async_pre_call_deployment_hook(kwargs, CallTypes.asearch)
    await callback.async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)
