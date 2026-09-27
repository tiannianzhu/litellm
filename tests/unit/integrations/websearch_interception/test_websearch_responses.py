"""
Integration tests for WebSearch interception with the Responses API.

Tests that the websearch_interception callback intercepts litellm_web_search
tool calls returned by /v1/responses, executes the search server-side, and
builds a Responses-format follow-up request.
"""

import asyncio
import itertools
import json
from collections.abc import AsyncIterator, Mapping
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.integrations.custom_logger import (
    RESPONSES_AGENTIC_SURFACE,
)
from litellm.types.integrations.websearch_interception import RichWebSearchInput
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import CallTypes, LlmProviders


def _bridge_sse(delta: dict[str, object], finish_reason: str | None = None) -> bytes:
    payload: Final = {
        "id": "chatcmpl-fixture",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _bridge_usage(prompt_tokens: int, completion_tokens: int) -> bytes:
    payload: Final = {
        "id": "chatcmpl-fixture",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


class _BridgeGatedStream(httpx.AsyncByteStream):
    def __init__(self, first: bytes, rest: bytes, release: asyncio.Event) -> None:
        self.first = first
        self.rest = rest
        self.release = release
        self.closed = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.first
        await self.release.wait()
        yield self.rest

    async def aclose(self) -> None:
        self.closed.set()


class _BridgeSearchLogger(WebSearchInterceptionLogger):
    def __init__(self) -> None:
        super().__init__(enabled_providers=[LlmProviders.HOSTED_VLLM])
        self.queries: asyncio.Queue[str] = asyncio.Queue()

    async def _execute_search(
        self,
        query: str,
        kwargs: Mapping[str, object] | None = None,
        rich: RichWebSearchInput | None = None,
    ) -> tuple[str, SearchResponse | None]:
        self.queries.put_nowait(query)
        return f"Result for {query}", None


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
async def test_hosted_vllm_responses_web_search_preserves_custom_tool_stream(
    monkeypatch: pytest.MonkeyPatch,
):
    tool_input: Final = "const result = await tools.exec_command({ cmd: 'true' });"
    response_body: Final = {
        "id": "chatcmpl_fixture",
        "object": "chat.completion",
        "created_at": 1,
        "model": "fixture",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_fixture_exec",
                            "type": "function",
                            "function": {"name": "exec", "arguments": json.dumps({"content": tool_input})},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
    }
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                _bridge_sse(
                    {
                        "tool_calls": [
                            {
                                **response_body["choices"][0]["message"]["tool_calls"][0],
                                "index": 0,
                            }
                        ]
                    }
                )
                + _bridge_sse({}, "tool_calls")
                + _bridge_usage(2, 3)
                + b"data: [DONE]\n\n"
            ),
        )

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
            use_chat_completions_api=True,
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
    assert upstream["stream"] is True
    assert [tool["function"]["name"] for tool in upstream["tools"]] == ["exec", "litellm_web_search"]
    event_types: Final = [
        getattr(getattr(event, "type", None), "value", getattr(event, "type", None)) for event in events
    ]
    assert (
        event_types.index("response.custom_tool_call_input.delta")
        < event_types.index("response.custom_tool_call_input.done")
        < event_types.index("response.output_item.done")
        < event_types.index("response.completed")
    )
    sequence_numbers: Final = [event.sequence_number for event in events]
    assert sequence_numbers == list(range(sequence_numbers[0], sequence_numbers[0] + len(events)))
    assert events[-1].model_dump()["sequence_number"] == sequence_numbers[-1]
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


@pytest.mark.asyncio
async def test_hosted_vllm_responses_bridge_yields_text_before_upstream_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release: Final = asyncio.Event()
    requests: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()
    logger: Final = _BridgeSearchLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_BridgeGatedStream(
                _bridge_sse({"content": "early"}),
                _bridge_sse({"content": " text"}) + _bridge_sse({}, "stop") + _bridge_usage(2, 3) + b"data: [DONE]\n\n",
                release,
            ),
        )

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        stream: Final = await litellm.aresponses(
            model="hosted_vllm/test-model",
            input="Say early text",
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            stream=True,
            use_chat_completions_api=True,
            tools=[{"type": "web_search"}],
            client=handler,
        )

        async def first_text_delta() -> str:
            async for event in stream:
                if getattr(event.type, "value", event.type) == "response.output_text.delta":
                    return event.delta
            raise AssertionError("The response ended without a text delta")

        try:
            first: Final = await asyncio.wait_for(first_text_delta(), timeout=2)
            assert first == "early"
            assert not release.is_set()
        finally:
            release.set()
        remaining: Final = [event async for event in stream]
    finally:
        release.set()
        await handler.client.aclose()

    request: Final = await asyncio.wait_for(requests.get(), timeout=2)
    assert request["stream"] is True
    assert requests.empty()
    assert logger.queries.empty()
    text_deltas: Final = [
        event.delta for event in remaining if getattr(event.type, "value", event.type) == "response.output_text.delta"
    ]
    assert "".join((first, *text_deltas)) == "early text"
    completed: Final = [
        event.response for event in remaining if getattr(event.type, "value", event.type) == "response.completed"
    ]
    assert len(completed) == 1
    assert completed[0].usage.total_tokens == 5


@pytest.mark.asyncio
async def test_hosted_vllm_responses_bridge_close_closes_upstream_before_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release: Final = asyncio.Event()
    raw_stream: Final = _BridgeGatedStream(
        _bridge_sse({"content": "partial"}),
        _bridge_sse({}, "stop") + b"data: [DONE]\n\n",
        release,
    )
    logger: Final = _BridgeSearchLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=raw_stream)

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        stream: Final = await litellm.aresponses(
            model="hosted_vllm/test-model",
            input="Stop after partial",
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            stream=True,
            use_chat_completions_api=True,
            tools=[{"type": "web_search"}],
            client=handler,
        )

        async def first_text_delta() -> str:
            async for event in stream:
                if getattr(event.type, "value", event.type) == "response.output_text.delta":
                    return event.delta
            raise AssertionError("The response ended without a text delta")

        assert await asyncio.wait_for(first_text_delta(), timeout=2) == "partial"
        await asyncio.wait_for(stream.aclose(), timeout=2)
        assert raw_stream.closed.is_set()
        assert not release.is_set()
    finally:
        release.set()
        await handler.client.aclose()

    assert logger.queries.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds", (1, 2))
@pytest.mark.parametrize("client_tool", (False, True))
async def test_hosted_vllm_responses_bridge_emits_completed_searches_across_rounds(
    monkeypatch: pytest.MonkeyPatch,
    rounds: int,
    client_tool: bool,
) -> None:
    call_number: Final = itertools.count()
    requests: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()
    logger: Final = _BridgeSearchLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    final_reply: Final = (
        _bridge_sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_client",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ]
            }
        )
        + _bridge_sse({}, "tool_calls")
        if client_tool
        else _bridge_sse({"content": "final answer"}) + _bridge_sse({}, "stop")
    )

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(json.loads(request.content))
        index: Final = next(call_number)
        body: Final = (
            (
                _bridge_sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_fixture",
                                "type": "function",
                                "function": {
                                    "name": "litellm_web_search",
                                    "arguments": json.dumps({"query": f"query {index}"}),
                                },
                            }
                        ]
                    }
                )
                + _bridge_sse({}, "tool_calls")
                if index < rounds
                else final_reply
            )
            + _bridge_usage(2, 3)
            + b"data: [DONE]\n\n"
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        stream: Final = await litellm.aresponses(
            model="hosted_vllm/test-model",
            input="Search before answering",
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            stream=True,
            use_chat_completions_api=True,
            tools=[
                {"type": "web_search"},
                {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
            ],
            client=handler,
        )
        events: Final = [event async for event in stream]
    finally:
        await handler.client.aclose()

    sent: Final = tuple(requests.get_nowait() for _ in range(requests.qsize()))
    assert len(sent) == rounds + 1
    assert all(request["stream"] is True for request in sent)
    assert tuple(logger.queries.get_nowait() for _ in range(logger.queries.qsize())) == tuple(
        f"query {index}" for index in range(rounds)
    )
    event_types: Final = [getattr(event.type, "value", event.type) for event in events]
    assert event_types.count("response.completed") == 1
    completed: Final = next(
        event.response for event in events if getattr(event.type, "value", event.type) == "response.completed"
    )
    web_search_calls: Final = [item for item in completed.output if item.type == "web_search_call"]
    assert len(web_search_calls) == rounds
    assert len({item.id for item in web_search_calls}) == rounds
    assert [item.action.query for item in web_search_calls] == [f"query {index}" for index in range(rounds)]
    done_searches: Final = [
        event.item
        for event in events
        if getattr(event.type, "value", event.type) == "response.output_item.done"
        and event.item.type == "web_search_call"
    ]
    assert [item.id for item in done_searches] == [item.id for item in web_search_calls]
    assert [item.action["query"] for item in done_searches] == [item.action.query for item in web_search_calls]
    client_calls: Final = [item for item in completed.output if item.type == "function_call"]
    assert [(item.call_id, item.name, item.arguments) for item in client_calls] == (
        [("call_client", "get_weather", '{"city":"Paris"}')] if client_tool else []
    )
    assert completed.usage.total_tokens == 5 * (rounds + 1)
    assert (
        "".join(
            event.delta for event in events if getattr(event.type, "value", event.type) == "response.output_text.delta"
        )
        == ("" if client_tool else "final answer")
    )


@pytest.mark.asyncio
async def test_hosted_vllm_responses_replay_completed_search_and_client_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_number: Final = itertools.count()
    requests: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()
    logger: Final = _BridgeSearchLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(json.loads(request.content))
        index: Final = next(request_number)
        if index == 0:
            body: Final = (
                _bridge_sse(
                    {
                        "content": "I will search, then run the client tool.",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_search",
                                "type": "function",
                                "function": {
                                    "name": "litellm_web_search",
                                    "arguments": json.dumps({"query": "latest ai news"}),
                                },
                            },
                            {
                                "index": 1,
                                "id": "call_exec",
                                "type": "function",
                                "function": {"name": "exec", "arguments": '{"command":"pwd"}'},
                            },
                        ],
                    }
                )
                + _bridge_sse({}, "tool_calls")
                + _bridge_usage(2, 3)
                + b"data: [DONE]\n\n"
            )
        else:
            body = (
                _bridge_sse({"content": "The search and client tool are complete."})
                + _bridge_sse({}, "stop")
                + _bridge_usage(3, 2)
                + b"data: [DONE]\n\n"
            )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    tools: Final = [
        {"type": "web_search"},
        {
            "type": "function",
            "name": "exec",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    ]
    try:
        first_stream: Final = await litellm.aresponses(
            model="hosted_vllm/test-model",
            input="Search and run the client tool.",
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            stream=True,
            use_chat_completions_api=True,
            tools=tools,
            client=handler,
        )
        first_events: Final = [event async for event in first_stream]
        first_completed: Final = next(
            event.response
            for event in first_events
            if getattr(event.type, "value", event.type) == "response.completed"
        )
        first_search: Final = next(item for item in first_completed.output if item.type == "web_search_call")
        first_exec: Final = next(
            item
            for item in first_completed.output
            if item.type == "function_call" and item.name == "exec"
        )

        second_stream: Final = await litellm.aresponses(
            model="hosted_vllm/test-model",
            input=[
                *(item.model_dump(exclude_none=True) for item in first_completed.output),
                {
                    "type": "function_call_output",
                    "call_id": first_exec.call_id,
                    "output": "exec succeeded",
                },
                {"role": "user", "content": "Summarize the results."},
            ],
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            stream=True,
            use_chat_completions_api=True,
            tools=tools,
            client=handler,
        )
        second_events: Final = [event async for event in second_stream]
    finally:
        await handler.client.aclose()

    sent: Final = tuple(requests.get_nowait() for _ in range(requests.qsize()))
    assert len(sent) == 2
    assert tuple(logger.queries.get_nowait() for _ in range(logger.queries.qsize())) == ("latest ai news",)

    replayed_messages: Final = sent[1]["messages"]
    assert isinstance(replayed_messages, list)
    assistant_messages: Final = [message for message in replayed_messages if message.get("role") == "assistant"]
    replayed_calls: Final = [call for message in assistant_messages for call in message.get("tool_calls", [])]
    search_call: Final = next(call for call in replayed_calls if call["function"]["name"] == "litellm_web_search")
    exec_call: Final = next(call for call in replayed_calls if call["function"]["name"] == "exec")
    assert json.loads(search_call["function"]["arguments"]) == {"query": "latest ai news"}
    assert search_call["id"] == first_search.id
    assert exec_call["id"] == first_exec.call_id
    assert all("Hosted web search:" not in str(message.get("content")) for message in assistant_messages)
    assert any("I will search, then run the client tool." in str(message.get("content")) for message in assistant_messages)

    request_tools: Final = sent[1]["tools"]
    assert isinstance(request_tools, list)
    replayed_search_tool: Final = next(
        tool["function"] for tool in request_tools if tool["function"]["name"] == "litellm_web_search"
    )
    assert replayed_search_tool["parameters"]["properties"]["query"]["type"] == "string"
    assert "query" in replayed_search_tool["parameters"]["required"]

    tool_results: Final = [message for message in replayed_messages if message.get("role") == "tool"]
    result_by_call_id: Final = {message.get("tool_call_id"): message.get("content") for message in tool_results}
    assert first_search.id in result_by_call_id
    assert "Result for latest ai news" in str(result_by_call_id[first_search.id])
    assert result_by_call_id[first_exec.call_id] == "exec succeeded"
    assert any(
        message.get("role") == "user" and message.get("content") == "Summarize the results."
        for message in replayed_messages
    )
    assert any(
        getattr(event.type, "value", event.type) == "response.completed" for event in second_events
    )
