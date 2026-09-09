"""
Test that responses() API does not create duplicate spend logs.

This test verifies the fix for issue #15740 where kwargs.pop() was removing
the logging object before passing kwargs to internal acompletion() calls,
causing duplicate spend log entries for non-OpenAI providers.
"""

import asyncio
import datetime
import json
from typing import Final, cast

import httpx
import pytest

import litellm
from litellm import Router
from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.integrations.custom_logger import AgenticLoopPlan, AgenticLoopRequestPatch
from litellm.types.utils import ModelResponse


def test_logging_object_not_popped():
    """
    Test that litellm_logging_obj is not popped from kwargs.

    This is a regression test for issue #15740. The bug was using
    kwargs.pop() which removed the logging object, causing duplicate
    spend logs for non-OpenAI providers.
    """
    import inspect

    from litellm.responses import main as responses_module

    # Get the source code of the responses function
    source = inspect.getsource(responses_module.responses)

    # Check that .pop("litellm_logging_obj") is NOT used
    # The bug was using kwargs.pop("litellm_logging_obj") which removes it
    assert 'kwargs.pop("litellm_logging_obj")' not in source, (
        "FAIL: Found kwargs.pop('litellm_logging_obj') in responses() function. "
        "This causes duplicate spend logs. Use kwargs.get('litellm_logging_obj') instead."
    )

    # Check that .get("litellm_logging_obj") IS used
    assert 'kwargs.get("litellm_logging_obj")' in source, (
        "FAIL: Expected kwargs.get('litellm_logging_obj') but not found. "
        "The logging object must be accessed with .get() not .pop() to prevent duplication."
    )


@pytest.mark.asyncio
async def test_async_no_duplicate_spend_logs():
    """
    Test that spend logs are only created once, not duplicated.

    This integration test verifies the fix by using a custom logger
    that counts log_success_event calls for a specific request ID.
    Before the fix, it would be called twice for non-OpenAI providers.
    """
    import uuid

    # Generate a unique ID to track only this test's request
    test_request_id = f"test-no-dup-{uuid.uuid4()}"

    # Create a custom logger to count log_success_event calls for our specific request
    class SpendLogCounter(CustomLogger):
        def __init__(self, tracking_id: str):
            super().__init__()
            self.tracking_id = tracking_id
            self.log_count = 0

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            # Only count logs for our specific test request
            litellm_call_id = kwargs.get("litellm_call_id", "")
            if litellm_call_id == self.tracking_id:
                self.log_count += 1

    spend_logger = SpendLogCounter(tracking_id=test_request_id)

    # Save original callbacks and append our logger (don't replace to avoid affecting other tests)
    original_callbacks = litellm.callbacks.copy() if litellm.callbacks else []
    litellm.callbacks = original_callbacks + [spend_logger]

    try:
        # Call responses API with Anthropic model using mock_response
        # Pass our unique ID as litellm_call_id to track this specific request
        response = await litellm.aresponses(
            model="anthropic/claude-3-7-sonnet-latest",
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                    "type": "message",
                }
            ],
            instructions="You are a helpful assistant.",
            mock_response="Hello! I'm doing well.",
            litellm_call_id=test_request_id,
        )

        # Yield to the event loop so the _client_async_logging_helper task
        # (scheduled via asyncio.create_task in the @client decorator) runs first
        # and initializes GLOBAL_LOGGING_WORKER on the current event loop.
        # Without this, flush() may block on a stale queue from a previous test's loop.
        await asyncio.sleep(0)

        # Wait for async logging to complete. Use a timeout so that if the
        # worker is on a stale event loop (common in CI), flush() doesn't hang
        # indefinitely — the queue.join() inside flush() would never resolve.
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        try:
            await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
        except asyncio.TimeoutError:
            pass
        await asyncio.sleep(0.5)

        # Verify that log_success_event was called exactly once for our request
        assert spend_logger.log_count == 1, (
            f"FAIL: log_success_event called {spend_logger.log_count} times instead of 1 "
            f"for request {test_request_id}. This indicates duplicate spend logs."
        )

    finally:
        # Restore original callbacks
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
@pytest.mark.parametrize("consume_stream", [True, False])
@pytest.mark.parametrize("search_used", [True, False])
async def test_aresponses_converted_web_search_stream_logs_completed_payload(
    monkeypatch: pytest.MonkeyPatch, consume_stream: bool, search_used: bool
):
    class SuccessPayloadRecorder(CustomLogger):
        def __init__(self):
            super().__init__()
            self.payloads: list[dict[str, object] | None] = []
            self.response_objects: list[ResponsesAPIResponse] = []
            self.finished = asyncio.Event()

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.payloads.append(kwargs.get("standard_logging_object"))
            self.response_objects.append(cast(ResponsesAPIResponse, response_obj))
            if len(self.payloads) == (2 if search_used else 1):
                self.finished.set()

    response_body: Final = {
        "id": "resp_web_search_stream",
        "object": "response",
        "created_at": 1,
        "model": "test-model",
        "output": [
            {
                "type": "message",
                "id": "msg_web_search_stream",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "complete answer", "annotations": []}],
            }
        ],
        "status": "completed",
        "usage": {
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    }
    search_response_body: Final = {
        **response_body,
        "id": "resp_search_round",
        "output": [
            {
                "type": "function_call",
                "name": LITELLM_WEB_SEARCH_TOOL_NAME,
                "call_id": "search_call",
                "arguments": '{"query":"fixture"}',
                "status": "completed",
            }
        ],
        "usage": {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
    }

    class FixtureSearch(WebSearchInterceptionLogger):
        async def async_build_responses_agentic_loop_plan(self, **kwargs: object) -> AgenticLoopPlan:
            return AgenticLoopPlan(
                run_agentic_loop=True,
                request_patch=AgenticLoopRequestPatch(
                    model="hosted_vllm/test-model",
                    messages=[{"role": "user", "content": "fixture search result"}],
                    kwargs={
                        "client": client,
                        "api_base": "http://test.invalid/v1",
                        "litellm_session_id": "web-search-session",
                    },
                ),
            )

    recorder: Final = SuccessPayloadRecorder()
    original_callbacks: Final = litellm.callbacks
    pricing: Final = {
        "litellm_provider": "hosted_vllm",
        "input_cost_per_token": 0.01,
        "output_cost_per_token": 0.1,
    }
    monkeypatch.setitem(litellm.model_cost, "test-model", pricing)
    monkeypatch.setitem(litellm.model_cost, "hosted_vllm/test-model", pricing)
    litellm.callbacks = [
        FixtureSearch(enabled_providers=["hosted_vllm"]),
        recorder,
    ]

    try:

        def respond(request: httpx.Request) -> httpx.Response:
            first_round: Final = "fixture search result" not in request.content.decode()
            return httpx.Response(
                200, json=search_response_body if search_used and first_round else response_body, request=request
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as upstream:
            client: Final = AsyncHTTPHandler()
            await client.close()
            client.client = upstream
            response = await litellm.aresponses(
                model="hosted_vllm/test-model",
                input="hello",
                api_base="http://test.invalid/v1",
                api_key="test-key",
                stream=True,
                tools=[{"type": "web_search"}],
                litellm_session_id="web-search-session",
                client=client,
            )
            terminal_response: Final = response.completed_response.response
            assert terminal_response.usage.output_tokens_details.reasoning_tokens == 2
            assert terminal_response.output[0].content[0].text == "complete answer"
            if consume_stream:
                async for _ in response:
                    pass

        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        await asyncio.wait_for(recorder.finished.wait(), timeout=5)
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5)
    finally:
        litellm.callbacks = original_callbacks

    assert len(recorder.payloads) == (2 if search_used else 1), [
        (p["id"], p["prompt_tokens"]) for p in recorder.payloads if p
    ]
    assert len({p["id"] for p in recorder.payloads if p is not None}) == len(recorder.payloads)
    assert sum(float(p["response_cost"]) for p in recorder.payloads if p is not None) == pytest.approx(
        0.96 if search_used else 0.32
    )
    payload: Final = next(p for p in recorder.payloads if p is not None and p["prompt_tokens"] == 2)
    assert isinstance(payload, dict)
    assert payload["prompt_tokens"] == 2
    assert payload["completion_tokens"] == 3
    assert payload["total_tokens"] == 5
    assert payload["response_cost"] == pytest.approx(0.32)
    assert payload["call_type"] == "aresponses"
    assert payload["session_id"] == "web-search-session"
    logged_response: Final = next(r for r in recorder.response_objects if r.usage.prompt_tokens == 2)
    assert logged_response.usage.completion_tokens_details.reasoning_tokens == 2
    assert logged_response.output[0].content[0].text == "complete answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("consume_stream", [True, False])
async def test_router_responses_custom_tool_bridge_logs_before_stream_consumption(
    monkeypatch: pytest.MonkeyPatch, consume_stream: bool
):
    class SuccessPayloadRecorder(CustomLogger):
        def __init__(self):
            super().__init__()
            self.payloads: list[dict[str, object] | None] = []
            self.response_objects: list[ModelResponse] = []
            self.finished = asyncio.Event()

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.payloads.append(kwargs.get("standard_logging_object"))
            self.response_objects.append(cast(ModelResponse, response_obj))
            self.finished.set()

    response_body: Final = {
        "id": "chatcmpl-custom-tool-bridge",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "complete answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }
    recorder: Final = SuccessPayloadRecorder()
    original_callbacks: Final = litellm.callbacks
    pricing: Final = {
        "litellm_provider": "hosted_vllm",
        "input_cost_per_token": 0.01,
        "output_cost_per_token": 0.1,
    }
    monkeypatch.setitem(litellm.model_cost, "test-model", pricing)
    monkeypatch.setitem(litellm.model_cost, "hosted_vllm/test-model", pricing)
    litellm.callbacks = [
        WebSearchInterceptionLogger(enabled_providers=["hosted_vllm"]),
        recorder,
    ]

    try:
        logging_obj, request_data = litellm.utils.function_setup(
            original_function="aresponses",
            rules_obj=litellm.utils.Rules(),
            start_time=datetime.datetime.now(),
            model="test-model-group",
            input="hello",
            stream=True,
            tools=[
                {"type": "web_search"},
                {"type": "custom", "name": "apply_patch", "format": {"type": "text"}},
            ],
            litellm_call_id="router-custom-tool-bridge",
        )
        request_data["litellm_logging_obj"] = logging_obj
        router: Final = Router(
            model_list=[
                {
                    "model_name": "test-model-group",
                    "litellm_params": {
                        "model": "hosted_vllm/test-model",
                        "api_base": "http://test.invalid/v1",
                        "api_key": "test-key",
                        "drop_params": True,
                    },
                    "model_info": {"id": "test-deployment", "supported_endpoints": ["/v1/messages"]},
                }
            ],
            num_retries=0,
        )
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=response_body, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as upstream:
            client: Final = AsyncHTTPHandler()
            await client.close()
            client.client = upstream
            router.model_list[0]["litellm_params"]["client"] = client

            response = await router.aresponses(**request_data)

            await asyncio.wait_for(recorder.finished.wait(), timeout=5)
            if consume_stream:
                stream_events: Final = [event async for event in response]
                terminal_response: Final = getattr(stream_events[-1], "response", None)
                assert isinstance(terminal_response, ResponsesAPIResponse)
                assert terminal_response.output[0].content[0].text == "complete answer"

        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5)
    finally:
        litellm.callbacks = original_callbacks

    assert len(requests) == 1
    assert requests[0].url.path.endswith("/chat/completions")
    assert json.loads(requests[0].content)["stream"] is False
    assert len(recorder.payloads) == 1
    payload: Final = recorder.payloads[0]
    assert payload is not None
    assert payload["id"] == "chatcmpl-custom-tool-bridge"
    assert payload["prompt_tokens"] == 2
    assert payload["completion_tokens"] == 3
    assert payload["total_tokens"] == 5
    assert payload["response_cost"] == pytest.approx(0.32)
    assert payload["call_type"] == "aresponses"
    assert recorder.response_objects[0].choices[0].message.content == "complete answer"
