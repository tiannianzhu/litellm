"""
Integration tests for WebSearch interception with the Responses API.

Tests that the websearch_interception callback intercepts litellm_web_search
tool calls returned by /v1/responses, executes the search server-side, and
builds a Responses-format follow-up request.
"""

import json
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.integrations.custom_logger import (
    RESPONSES_AGENTIC_SURFACE,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import CallTypes, LlmProviders


def _responses_output_with_web_search(call_id: str = "fc_1", query: str = "latest ai news"):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="litellm_web_search",
                call_id=call_id,
                arguments='{"query": "%s"}' % query,
            )
        ]
    )


@pytest.mark.asyncio
async def test_responses_hook_detects_function_call():
    """async_should_run_responses_agentic_loop detects a litellm_web_search function_call."""
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    should_run, tools_dict = await logger.async_should_run_responses_agentic_loop(
        response=_responses_output_with_web_search(),
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the latest AI news?"}],
        tools=[{"type": "function", "name": "litellm_web_search"}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is True
    assert tools_dict["response_format"] == "responses"
    assert len(tools_dict["tool_calls"]) == 1
    assert tools_dict["tool_calls"][0]["name"] == "litellm_web_search"
    assert tools_dict["tool_calls"][0]["call_id"] == "fc_1"
    assert tools_dict["tool_calls"][0]["input"] == {"query": "latest ai news"}


@pytest.mark.asyncio
async def test_responses_hook_not_triggered_without_tool():
    """No web search tool in the request -> hook must not run."""
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    should_run, tools_dict = await logger.async_should_run_responses_agentic_loop(
        response=_responses_output_with_web_search(),
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "name": "get_weather"}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_responses_hook_not_triggered_for_disabled_provider():
    """Provider not in enabled_providers -> hook must not run."""
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK])

    should_run, tools_dict = await logger.async_should_run_responses_agentic_loop(
        response=_responses_output_with_web_search(),
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "name": "litellm_web_search"}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_responses_hook_ignores_non_websearch_function_call():
    """A function_call for a different tool must not be intercepted."""
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])
    response = SimpleNamespace(
        output=[SimpleNamespace(type="function_call", name="get_weather", call_id="c1", arguments="{}")]
    )

    should_run, tools_dict = await logger.async_should_run_responses_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "name": "litellm_web_search"}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_responses_hook_ignores_bare_web_search_function_call():
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])
    response = SimpleNamespace(
        output=[SimpleNamespace(type="function_call", name="web_search", call_id="c1", arguments="{}")]
    )

    should_run, tools_dict = await logger.async_should_run_responses_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "name": "litellm_web_search"}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_surface_marker_routes_should_run_to_responses_branch():
    """async_should_run_agentic_loop must dispatch to the responses branch when the
    surface marker says responses.

    Without the marker the default anthropic branch runs and never detects the
    Responses-format function_call, so interception silently no-ops on /v1/responses.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    should_run, tools_dict = await logger.async_should_run_agentic_loop(
        response=_responses_output_with_web_search(),
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the latest AI news?"}],
        tools=[{"type": "function", "name": "litellm_web_search"}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={"_agentic_loop_api_surface": RESPONSES_AGENTIC_SURFACE},
    )

    assert should_run is True
    assert tools_dict["response_format"] == "responses"


@pytest.mark.asyncio
async def test_default_branch_does_not_detect_responses_output():
    """Regression guard: the default (anthropic) branch must not detect a
    Responses-format function_call, proving the responses branch is required.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    should_run, tools_dict = await logger.async_should_run_agentic_loop(
        response=_responses_output_with_web_search(),
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "name": "litellm_web_search"}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False


@pytest.mark.asyncio
async def test_build_responses_plan_produces_responses_input():
    """async_build_responses_agentic_loop_plan builds a Responses-format follow-up:
    the user input followed by function_call + function_call_output items, with
    the web search tool preserved and tool_choice stripped.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    tools_dict = {
        "tool_calls": [
            {
                "id": "fc_1",
                "call_id": "fc_1",
                "type": "function_call",
                "name": "litellm_web_search",
                "arguments": '{"query": "latest ai news"}',
                "input": {"query": "latest ai news"},
            }
        ],
        "tool_type": "websearch",
        "provider": "openai",
        "response_format": "responses",
    }

    with patch.object(
        logger,
        "_execute_search",
        new=AsyncMock(return_value=("OpenAI shipped a new model", None)),
    ):
        plan = await logger.async_build_responses_agentic_loop_plan(
            tools=tools_dict,
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's the latest AI news?"}],
            response=_responses_output_with_web_search(),
            optional_params={
                "tools": [{"type": "function", "name": "litellm_web_search"}],
                "tool_choice": {"type": "function", "name": "litellm_web_search"},
            },
            logging_obj=MagicMock(),
            stream=False,
            kwargs={
                "custom_llm_provider": "openai",
                "_agentic_loop_api_surface": RESPONSES_AGENTIC_SURFACE,
            },
        )

    assert plan.run_agentic_loop is True
    patch_obj = plan.request_patch
    assert patch_obj is not None
    input_items = patch_obj.messages
    assert input_items is not None

    assert input_items[0] == {"role": "user", "content": "What's the latest AI news?"}
    assert input_items[1] == {
        "type": "function_call",
        "call_id": "fc_1",
        "name": "litellm_web_search",
        "arguments": '{"query": "latest ai news"}',
    }
    assert input_items[2] == {
        "type": "function_call_output",
        "call_id": "fc_1",
        "output": "OpenAI shipped a new model",
    }

    assert patch_obj.tools == [{"type": "function", "name": "litellm_web_search"}]
    assert "tool_choice" not in patch_obj.optional_params
    assert "_agentic_loop_api_surface" not in patch_obj.kwargs
    assert patch_obj.model == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_deployment_hook_converts_native_responses_web_search_tool():
    """async_pre_call_deployment_hook converts a native Responses web_search tool
    into the flat litellm_web_search function tool (Responses shape, not the
    nested Chat Completions {"function": {...}} wrapper).
    """
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    result = await logger.async_pre_call_deployment_hook(
        kwargs={
            "model": "gpt-4o",
            "custom_llm_provider": "openai",
            "tools": [{"type": "web_search"}],
        },
        call_type=CallTypes.aresponses,
    )

    assert result is not None
    converted_tools = result["tools"]
    assert len(converted_tools) == 1
    tool = converted_tools[0]
    assert tool["type"] == "function"
    assert tool["name"] == "litellm_web_search"
    assert "function" not in tool
    assert tool["parameters"]["required"] == ["query"]


@pytest.mark.asyncio
async def test_deployment_hook_responses_returns_none_without_web_search():
    """No web search tool in a responses request -> deployment hook makes no change."""
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    result = await logger.async_pre_call_deployment_hook(
        kwargs={
            "model": "gpt-4o",
            "custom_llm_provider": "openai",
            "tools": [{"type": "function", "name": "get_weather"}],
        },
        call_type=CallTypes.aresponses,
    )

    assert result is None


@pytest.mark.asyncio
async def test_deployment_hook_responses_converts_stream_to_non_stream():
    """Streaming responses requests are converted to non-streaming so the agentic
    loop can run, and flagged for re-wrapping afterwards.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    result = await logger.async_pre_call_deployment_hook(
        kwargs={
            "model": "gpt-4o",
            "custom_llm_provider": "openai",
            "tools": [{"type": "web_search_preview"}],
            "stream": True,
        },
        call_type=CallTypes.aresponses,
    )

    assert result is not None
    assert result["stream"] is False
    assert result["_websearch_interception_converted_stream"] is True


@pytest.mark.asyncio
async def test_hosted_vllm_responses_web_search_interceptor_rewraps_custom_tool_stream(
    monkeypatch: pytest.MonkeyPatch,
):
    tool_input: Final = "const result = await tools.exec_command({ cmd: 'true' });"
    function_call: Final = {
        "type": "function_call",
        "id": "fc_fixture_exec",
        "call_id": "call_fixture_exec",
        "name": "exec",
        "arguments": json.dumps({"content": tool_input}),
        "status": "completed",
    }
    response_body: Final = {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 1,
        "model": "fixture",
        "status": "completed",
        "output": [function_call],
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "parallel_tool_calls": True,
        "temperature": None,
        "tool_choice": "none",
        "tools": [],
        "top_p": None,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": None,
        "text": {},
        "truncation": None,
        "user": None,
        "store": False,
        "usage": {
            "input_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 5,
        },
    }
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response_body)

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    interceptor: Final = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.HOSTED_VLLM])
    monkeypatch.setattr(litellm, "callbacks", [interceptor])
    try:
        stream: Final = await litellm.aresponses(
            model="hosted_vllm/fixture",
            input="Return the synthetic tool call.",
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            stream=True,
            tool_choice="none",
            tools=[
                {
                    "type": "custom",
                    "name": "exec",
                    "format": {"type": "grammar", "syntax": "lark", "definition": "start: /[\\s\\S]+/"},
                },
                {"type": "web_search"},
            ],
            client=handler,
        )
        events: Final = [event async for event in stream]
    finally:
        await handler.client.aclose()

    assert len(sent) == 1
    upstream: Final = sent[0]
    assert upstream["stream"] is False
    assert [tool["name"] for tool in upstream["tools"]] == ["exec", "litellm_web_search"]
    event_types: Final = [
        getattr(getattr(event, "type", None), "value", getattr(event, "type", None)) for event in events
    ]
    assert (
        event_types.index("response.custom_tool_call_input.delta")
        < event_types.index("response.custom_tool_call_input.done")
        < event_types.index("response.output_item.done")
        < event_types.index("response.completed")
    )
    assert [event.sequence_number for event in events] == list(range(len(events)))
    added: Final = next(
        event for event in events if getattr(event.type, "value", event.type) == "response.output_item.added"
    )
    assert added.item.type == "custom_tool_call"
    assert added.item.input == ""
    assert added.item.status == "in_progress"
    completed: Final = next(
        event.response for event in events if getattr(event.type, "value", event.type) == "response.completed"
    )
    assert isinstance(completed, ResponsesAPIResponse)
    assert completed.output[0].type == "custom_tool_call"
    assert completed.output[0].call_id == "call_fixture_exec"
    assert completed.output[0].input == tool_input
    assert completed.usage.total_tokens == 5
