"""
Test search API logging and cost tracking in proxy.

Tests that search API requests are properly logged to LiteLLM_SpendLogs
with correct fields populated (call_type, model, custom_llm_provider, 
model_group, spend, etc.)
"""

import asyncio
import os
import time
from datetime import datetime
from typing import Final
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm
from litellm import Router
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
from litellm.proxy.spend_tracking.spend_management_endpoints import view_spend_logs
from litellm.proxy.utils import ProxyLogging, hash_token, update_spend
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from tests._master_key import MASTER_KEY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "registered_price", [None, 0.007, 0.0],
)
async def test_bocha_search_registers_price_and_logs_once_after_catalog_reload(
    monkeypatch: pytest.MonkeyPatch, registered_price: float | None
) -> None:
    from litellm import utils as litellm_utils
    from litellm.litellm_core_utils.get_model_cost_map import adopt_model_cost_map
    from litellm.llms.bocha.search.transformation import BochaSearchConfig
    from litellm.llms.custom_httpx import llm_http_handler
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    class SearchRecorder(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.records: asyncio.Queue[float] = asyncio.Queue()

        async def async_log_success_event(
            self,
            kwargs: dict[str, object],
            response_obj: object,
            start_time: datetime,
            end_time: datetime,
        ) -> None:
            if kwargs.get("call_type") != "asearch":
                return
            cost: Final = kwargs.get("response_cost")
            assert isinstance(cost, (float, int))
            self.records.put_nowait(float(cost))

    async def respond(request: httpx.Request) -> httpx.Response:
        assert "input_cost_per_query" not in request.content.decode()
        return httpx.Response(200, json={"code": 200, "data": {"webPages": {"value": []}}}, request=request)

    tool_params: Final[dict[str, str | float]] = {
        "search_provider": "bocha",
        "api_key": "test-key",
    }
    router: Final = Router(model_list=[], search_tools=[{"search_tool_name": "test-search", "litellm_params": tool_params}])
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    recorder: Final = SearchRecorder()
    monkeypatch.setattr(llm_http_handler, "get_async_httpx_client", lambda **_: handler)
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.model_cost,
            "bocha/search": {"litellm_provider": "bocha", "mode": "search"},
        },
    )

    monkeypatch.setattr(litellm_utils, "_runtime_registered_model_cost", {})
    if registered_price is not None:
        litellm.register_model({"bocha/search": {"input_cost_per_query": registered_price}})
    BochaSearchConfig()
    expected_price: Final = litellm.model_cost["bocha/search"]["input_cost_per_query"]
    if registered_price is None:
        assert expected_price > 0
    else:
        assert expected_price == registered_price
    adopt_model_cost_map({key: value for key, value in litellm.model_cost.items() if key != "bocha/search"})
    assert litellm.model_cost["bocha/search"]["input_cost_per_query"] == expected_price

    try:
        await router.asearch(search_tool_name="test-search", query="query")
        logged_cost: Final = await asyncio.wait_for(recorder.records.get(), timeout=5)
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5)
        assert logged_cost == pytest.approx(expected_price)
        assert recorder.records.empty()
    finally:
        await handler.client.aclose()


@pytest.fixture
def prisma_client():
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_cli import append_query_params
    from litellm.proxy.utils import PrismaClient

    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    if database_url is None:
        pytest.skip("DATABASE_URL not set")

    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url

    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)

    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    proxy_server.litellm_proxy_budget_name = f"litellm-proxy-budget-{time.time()}"
    proxy_server.user_custom_key_generate = None

    return prisma_client


@pytest.mark.skip(reason="Requires reliable external DB connection (prisma).")
@pytest.mark.asyncio
async def test_search_api_logging_and_cost_tracking(prisma_client):
    """
    Test that search API requests are logged with correct fields and cost tracking.

    Verifies:
    1. Search request creates a spend log entry
    2. call_type is set to "asearch"
    3. model is set to search_tool_name
    4. custom_llm_provider is set correctly
    5. model_group is set to search_tool_name
    6. spend is calculated and logged
    """
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)
    await litellm.proxy.proxy_server.prisma_client.connect()

    # Setup router with search tool
    search_tool_name = "tavily-search"
    search_provider = "tavily"

    router = Router(model_list=[])
    router.search_tools = [
        {
            "search_tool_name": search_tool_name,
            "litellm_params": {
                "search_provider": search_provider,
            },
        }
    ]

    setattr(litellm.proxy.proxy_server, "llm_router", router)

    # Generate a test API key
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_fn,
    )
    from litellm.proxy._types import GenerateKeyRequest

    from litellm.proxy._types import LitellmUserRoles

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key=MASTER_KEY,
        user_id="test_user",
    )

    key_request = GenerateKeyRequest(models=[], duration=None)
    key_response = await generate_key_fn(
        data=key_request, user_api_key_dict=user_api_key_dict
    )
    generated_key = key_response.key
    user_id = key_response.user_id

    # Create mock search response
    mock_search_result = SearchResult(
        title="Test Result",
        url="https://example.com",
        snippet="Test snippet",
    )

    mock_search_response = SearchResponse(
        object="search",
        results=[mock_search_result],
    )

    # Mock the search function to return our mock response
    with patch("litellm.search.main.asearch", new_callable=AsyncMock) as mock_asearch:
        mock_asearch.return_value = mock_search_response

        # Setup proxy logging
        user_api_key_cache = DualCache()
        proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)
        setattr(litellm.proxy.proxy_server, "proxy_logging_obj", proxy_logging_obj)

        # Call the track_cost_callback directly to simulate what happens after a search
        proxy_db_logger = ProxyDBLogger()

        # Simulate the kwargs that would be passed from the search endpoint
        request_id = "search_test_123"
        kwargs = {
            "call_type": "asearch",
            "model": search_tool_name,
            "custom_llm_provider": search_provider,
            "litellm_call_id": request_id,  # Set request_id in kwargs
            "litellm_params": {
                "metadata": {
                    "user_api_key": hash_token(generated_key),
                    "user_api_key_user_id": user_id,
                    "model_group": search_tool_name,
                }
            },
            "metadata": {
                "user_api_key": hash_token(generated_key),
                "user_api_key_user_id": user_id,
                "model_group": search_tool_name,
            },
            "response_cost": 0.008,  # Mock cost for tavily search
        }

        # Set id on the response object
        mock_search_response.id = request_id

        await proxy_db_logger._PROXY_track_cost_callback(
            kwargs=kwargs,
            completion_response=mock_search_response,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        # Wait for async operations
        await asyncio.sleep(2)
        await update_spend(
            prisma_client=prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging_obj,
        )

        # Query spend logs
        spend_logs = await view_spend_logs(
            request_id=request_id,
            user_api_key_dict=UserAPIKeyAuth(api_key=generated_key),
        )

        # Verify spend log was created
        assert len(spend_logs) == 1, f"Expected 1 spend log, got {len(spend_logs)}"

        spend_log = spend_logs[0]

        # Verify all fields are populated correctly
        assert spend_log.request_id == request_id
        assert spend_log.call_type == "asearch"
        assert spend_log.model == search_tool_name
        assert spend_log.custom_llm_provider == search_provider
        assert spend_log.model_group == search_tool_name
        assert spend_log.spend == 0.008
        # API key should be hashed (either the generated key or the one from metadata)
        assert spend_log.api_key != ""  # Should be populated
        # Note: user field may be empty if not set in the request, but user_id should be in metadata
        assert (
            spend_log.metadata.get("user_api_key_user_id") == user_id
            or spend_log.user == user_id
        )

        print(f"✅ Search API logging test passed!")
        print(f"   - call_type: {spend_log.call_type}")
        print(f"   - model: {spend_log.model}")
        print(f"   - custom_llm_provider: {spend_log.custom_llm_provider}")
        print(f"   - model_group: {spend_log.model_group}")
        print(f"   - spend: {spend_log.spend}")
