"""
Integration tests for WebSearch interception with chat completions API.

Tests the end-to-end flow of websearch_interception callback with
litellm.acompletion() for transparent server-side web search execution.
"""

import asyncio
import itertools
import json
import os
from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletionChunk
from pydantic import ValidationError

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.integrations.websearch_interception import RichWebSearchInput
from litellm.types.utils import LlmProviders, ModelResponse


def _chat_sse(delta: dict[str, object], finish_reason: str | None = None, wire_id: str = "chatcmpl-fixture") -> bytes:
    payload: Final = {
        "id": wire_id,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _usage_sse(prompt_tokens: int, completion_tokens: int, wire_id: str = "chatcmpl-fixture") -> bytes:
    payload: Final = {
        "id": wire_id,
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


class _GatedSSEStream(httpx.AsyncByteStream):
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


class _RecordingWebSearchLogger(WebSearchInterceptionLogger):
    def __init__(self, failure: bool = False, max_agentic_loops: int | None = None) -> None:
        super().__init__(enabled_providers=[LlmProviders.HOSTED_VLLM], max_agentic_loops=max_agentic_loops)
        self.queries: asyncio.Queue[str] = asyncio.Queue()
        self.failure = failure

    async def _execute_search(
        self,
        query: str,
        kwargs: Mapping[str, object] | None = None,
        rich: RichWebSearchInput | None = None,
    ) -> tuple[str, SearchResponse | None]:
        self.queries.put_nowait(query)
        if self.failure:
            raise RuntimeError("fixture unavailable")
        return "fixture result", None


class _RoundObserver(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.rounds: asyncio.Queue[tuple[str, int, object, object, object]] = asyncio.Queue()

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        if isinstance(response_obj, ModelResponse) and response_obj.usage is not None:
            litellm_params: Final = kwargs.get("litellm_params")
            self.rounds.put_nowait(
                (
                    response_obj.id,
                    response_obj.usage.total_tokens,
                    litellm_params.get("metadata") if isinstance(litellm_params, Mapping) else None,
                    litellm_params.get("litellm_metadata") if isinstance(litellm_params, Mapping) else None,
                    kwargs.get("user"),
                )
            )


@pytest.mark.asyncio
async def test_declared_web_search_stream_yields_text_before_backend_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release: Final = asyncio.Event()
    requests: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()
    logger: Final = _RecordingWebSearchLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_GatedSSEStream(
                _chat_sse({"content": "Hello"}),
                _chat_sse({"content": " world"}) + _chat_sse({}, "stop") + _usage_sse(3, 2) + b"data: [DONE]\n\n",
                release,
            ),
        )

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        stream: Final = await litellm.acompletion(
            model="hosted_vllm/test-model",
            messages=[{"role": "user", "content": "Say hello"}],
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            tools=[{"type": "function", "function": {"name": "litellm_web_search"}}],
            stream=True,
            stream_options={"include_usage": True},
            client=handler,
        )
        try:
            first: Final = await asyncio.wait_for(anext(stream), timeout=2)
            assert first.choices[0].delta.content == "Hello"
            assert not release.is_set()
        finally:
            release.set()
        remaining: Final = [chunk async for chunk in stream]
    finally:
        release.set()
        await handler.client.aclose()

    request: Final = await asyncio.wait_for(requests.get(), timeout=2)
    assert request["stream"] is True
    assert requests.empty()
    assert logger.queries.empty()
    choices: Final = tuple(itertools.chain.from_iterable(chunk.choices for chunk in (first, *remaining)))
    assert "".join(choice.delta.content or "" for choice in choices) == "Hello world"
    assert [choice.finish_reason for choice in choices if choice.finish_reason] == ["stop"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_result"),
    ((False, "fixture result"), (True, "Search failed: fixture unavailable")),
)
@pytest.mark.parametrize("history_kind", ("text", "content_parts", "tool_calls"))
async def test_split_web_search_stream_executes_search_and_streams_followup(
    monkeypatch: pytest.MonkeyPatch,
    failure: bool,
    expected_result: str,
    history_kind: str,
) -> None:
    call_number: Final = itertools.count()
    requests: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()
    logger: Final = _RecordingWebSearchLogger(failure=failure)
    monkeypatch.setattr(litellm, "callbacks", [logger])
    first_body: Final = (
        _chat_sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_fixture",
                        "type": "function",
                        "function": {"name": "litellm_web_search", "arguments": '{"query":"fixture'},
                    }
                ]
            }
        )
        + _chat_sse({"tool_calls": [{"index": 0, "function": {"arguments": ' query"}'}}]})
        + _chat_sse({}, "tool_calls")
        + _usage_sse(2, 3)
        + b"data: [DONE]\n\n"
    )
    second_body: Final = (
        _chat_sse({"content": "The answer"}) + _chat_sse({}, "stop") + _usage_sse(4, 5) + b"data: [DONE]\n\n"
    )

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(json.loads(request.content))
        body: Final = first_body if next(call_number) == 0 else second_body
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    history: Final = (
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_prior",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_prior", "content": "Prior result"},
        ]
        if history_kind == "tool_calls"
        else []
    )
    messages: Final = [
        *history,
        {
            "role": "user",
            "content": (
                [{"type": "text", "text": "Find fixture"}, {"type": "text", "text": " query"}]
                if history_kind == "content_parts"
                else "Find fixture query"
            ),
        },
    ]
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        stream: Final = await litellm.acompletion(
            model="hosted_vllm/test-model",
            messages=messages,
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            tools=[{"type": "function", "function": {"name": "litellm_web_search"}}],
            stream=True,
            stream_options={"include_usage": True},
            client=handler,
        )
        chunks: Final = [chunk async for chunk in stream]
    finally:
        await handler.client.aclose()

    first_request: Final = await asyncio.wait_for(requests.get(), timeout=2)
    second_request: Final = await asyncio.wait_for(requests.get(), timeout=2)
    assert requests.empty()
    assert first_request["stream"] is True
    assert second_request["stream"] is True
    assert second_request["messages"][:-2] == first_request["messages"]
    assert second_request["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_fixture",
        "content": expected_result,
    }
    assert await asyncio.wait_for(logger.queries.get(), timeout=2) == "fixture query"
    assert logger.queries.empty()
    choices: Final = tuple(itertools.chain.from_iterable(chunk.choices for chunk in chunks))
    assert "".join(choice.delta.content or "" for choice in choices) == "The answer"
    assert all(not choice.delta.tool_calls for choice in choices)
    search_markers: Final = tuple(
        field["web_search_calls"]
        for choice in choices
        if (field := choice.delta.provider_specific_fields) and "web_search_calls" in field
    )
    assert len(search_markers) == 1
    assert search_markers[0][0]["action"]["query"] == "fixture query"
    assert [choice.finish_reason for choice in choices if choice.finish_reason] == ["stop"]
    usage: Final = [value for chunk in chunks if (value := getattr(chunk, "usage", None)) is not None]
    assert len(usage) == 1
    assert (usage[0].prompt_tokens, usage[0].completion_tokens, usage[0].total_tokens) == (6, 8, 14)


@pytest.mark.asyncio
async def test_web_search_stream_close_closes_upstream_without_search(monkeypatch: pytest.MonkeyPatch) -> None:
    release: Final = asyncio.Event()
    raw_stream: Final = _GatedSSEStream(
        _chat_sse({"content": "partial"}),
        _chat_sse({}, "stop") + b"data: [DONE]\n\n",
        release,
    )
    requests: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()
    logger: Final = _RecordingWebSearchLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=raw_stream)

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        stream: Final = await litellm.acompletion(
            model="hosted_vllm/test-model",
            messages=[{"role": "user", "content": "Stop after partial"}],
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            tools=[{"type": "function", "function": {"name": "litellm_web_search"}}],
            stream=True,
            client=handler,
        )
        first: Final = await asyncio.wait_for(anext(stream), timeout=2)
        assert first.choices[0].delta.content == "partial"
        await asyncio.wait_for(stream.aclose(), timeout=2)
    finally:
        release.set()
        await handler.client.aclose()

    assert raw_stream.closed.is_set()
    assert requests.qsize() == 1
    assert logger.queries.empty()


@pytest.mark.asyncio
async def test_incomplete_web_search_arguments_are_not_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()
    logger: Final = _RecordingWebSearchLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    body: Final = (
        _chat_sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_fixture",
                        "type": "function",
                        "function": {"name": "litellm_web_search", "arguments": '{"query":"unfinished'},
                    }
                ]
            }
        )
        + _chat_sse({}, "length")
        + b"data: [DONE]\n\n"
    )

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        stream: Final = await litellm.acompletion(
            model="hosted_vllm/test-model",
            messages=[{"role": "user", "content": "Find something"}],
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            tools=[{"type": "function", "function": {"name": "litellm_web_search"}}],
            stream=True,
            client=handler,
        )
        chunks: Final = [chunk async for chunk in stream]
    finally:
        await handler.client.aclose()

    choices: Final = tuple(itertools.chain.from_iterable(chunk.choices for chunk in chunks))
    assert [choice.finish_reason for choice in choices if choice.finish_reason] == ["length"]
    assert logger.queries.empty()
    assert requests.qsize() == 1


@pytest.mark.asyncio
async def test_malformed_web_search_arguments_fail_without_search(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()
    logger: Final = _RecordingWebSearchLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    body: Final = (
        _chat_sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_fixture",
                        "type": "function",
                        "function": {"name": "litellm_web_search", "arguments": '{"query":'},
                    }
                ]
            }
        )
        + _chat_sse({}, "tool_calls")
        + b"data: [DONE]\n\n"
    )

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        stream: Final = await litellm.acompletion(
            model="hosted_vllm/test-model",
            messages=[{"role": "user", "content": "Find something"}],
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            tools=[{"type": "function", "function": {"name": "litellm_web_search"}}],
            stream=True,
            client=handler,
        )
        with pytest.raises(ValidationError):
            _ = [chunk async for chunk in stream]
    finally:
        await handler.client.aclose()

    assert logger.queries.empty()
    assert requests.qsize() == 1


@pytest.mark.asyncio
async def test_mixed_web_search_and_client_tool_stream_keeps_client_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()
    logger: Final = _RecordingWebSearchLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    body: Final = (
        _chat_sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_search",
                        "type": "function",
                        "function": {"name": "litellm_web_search", "arguments": '{"query":"fixture query"}'},
                    },
                    {
                        "index": 1,
                        "id": "call_client",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    },
                ]
            }
        )
        + _chat_sse({}, "tool_calls")
        + b"data: [DONE]\n\n"
    )

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        stream: Final = await litellm.acompletion(
            model="hosted_vllm/test-model",
            messages=[{"role": "user", "content": "Search and check weather"}],
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            tools=[
                {"type": "function", "function": {"name": "litellm_web_search"}},
                {"type": "function", "function": {"name": "get_weather"}},
            ],
            stream=True,
            client=handler,
        )
        chunks: Final = [chunk async for chunk in stream]
    finally:
        await handler.client.aclose()

    choices: Final = tuple(itertools.chain.from_iterable(chunk.choices for chunk in chunks))
    visible_calls: Final = tuple(itertools.chain.from_iterable(choice.delta.tool_calls or () for choice in choices))
    assert [(call.id, call.function.name) for call in visible_calls] == [("call_client", "get_weather")]
    sdk_state: Final = ChatCompletionStreamState()
    for chunk in chunks:
        sdk_state.handle_chunk(ChatCompletionChunk.model_validate(chunk.model_dump(exclude_none=True)))
    sdk_calls: Final = sdk_state.get_final_completion().choices[0].message.tool_calls
    assert sdk_calls is not None
    assert [(call.id, call.function.name, call.function.arguments) for call in sdk_calls] == [
        ("call_client", "get_weather", "{}")
    ]
    search_markers: Final = tuple(
        field["web_search_calls"]
        for choice in choices
        if (field := choice.delta.provider_specific_fields) and "web_search_calls" in field
    )
    assert len(search_markers) == 1
    assert search_markers[0][0]["status"] == "completed"
    assert [choice.finish_reason for choice in choices if choice.finish_reason] == ["tool_calls"]
    assert await asyncio.wait_for(logger.queries.get(), timeout=2) == "fixture query"
    assert logger.queries.empty()
    assert requests.qsize() == 1


@pytest.mark.asyncio
async def test_web_search_stream_logs_each_backend_round_with_own_usage_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_number: Final = itertools.count()
    requests: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()
    logger: Final = _RecordingWebSearchLogger()
    observer: Final = _RoundObserver()
    monkeypatch.setattr(litellm, "callbacks", [logger, observer])

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(json.loads(request.content))
        first: Final = next(call_number) == 0
        wire_id: Final = "wire_search" if first else "wire_answer"
        body: Final = (
            (
                _chat_sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_search",
                                "type": "function",
                                "function": {"name": "litellm_web_search", "arguments": '{"query":"fixture query"}'},
                            }
                        ]
                    },
                    wire_id=wire_id,
                )
                + _chat_sse({}, "tool_calls", wire_id=wire_id)
                if first
                else _chat_sse({"content": "answer"}, wire_id=wire_id) + _chat_sse({}, "stop", wire_id=wire_id)
            )
            + _usage_sse(2, 3, wire_id=wire_id)
            + b"data: [DONE]\n\n"
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        stream: Final = await litellm.acompletion(
            model="hosted_vllm/test-model",
            messages=[{"role": "user", "content": "Find fixture query"}],
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            tools=[{"type": "function", "function": {"name": "litellm_web_search"}}],
            stream=True,
            stream_options={"include_usage": True},
            user="user_fixture",
            metadata={"session_id": "session_fixture"},
            client=handler,
        )
        chunks: Final = [chunk async for chunk in stream]
        observed: Final = (
            await asyncio.wait_for(observer.rounds.get(), timeout=3),
            await asyncio.wait_for(observer.rounds.get(), timeout=3),
        )
    finally:
        await handler.client.aclose()

    assert [(round_id, tokens) for round_id, tokens, *_ in observed] == [
        ("wire_search", 5),
        ("wire_answer", 5),
    ]
    assert observer.rounds.empty()
    assert all(
        (isinstance(metadata, Mapping) and metadata.get("session_id") == "session_fixture")
        or (isinstance(litellm_metadata, Mapping) and litellm_metadata.get("session_id") == "session_fixture")
        for _, _, metadata, litellm_metadata, _ in observed
    )
    assert all(user == "user_fixture" for _, _, _, _, user in observed)
    sent: Final = (requests.get_nowait(), requests.get_nowait())
    assert requests.empty()
    assert sent[1]["messages"][-1]["content"] == "fixture result"
    assert [value.total_tokens for chunk in chunks if (value := getattr(chunk, "usage", None)) is not None] == [10]


@pytest.mark.asyncio
async def test_web_search_stream_respects_callback_loop_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    call_number: Final = itertools.count()
    requests: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()
    logger: Final = _RecordingWebSearchLogger(max_agentic_loops=1)
    monkeypatch.setattr(litellm, "callbacks", [logger])

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(json.loads(request.content))
        index: Final = next(call_number)
        body: Final = (
            _chat_sse(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": "litellm_web_search",
                                "arguments": json.dumps({"query": f"query {index}"}),
                            },
                        }
                    ]
                }
            )
            + _chat_sse({}, "tool_calls")
            + b"data: [DONE]\n\n"
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        stream: Final = await litellm.acompletion(
            model="hosted_vllm/test-model",
            messages=[{"role": "user", "content": "Keep searching"}],
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            tools=[{"type": "function", "function": {"name": "litellm_web_search"}}],
            stream=True,
            client=handler,
        )
        with pytest.raises(ValueError, match="Exceeded max_agentic_loops=1"):
            _ = [chunk async for chunk in stream]
    finally:
        await handler.client.aclose()

    assert requests.qsize() == 2
    assert logger.queries.get_nowait() == "query 0"
    assert logger.queries.empty()


@pytest.fixture
def mock_search_response():
    """Mock search response from litellm.asearch()"""
    mock_response = MagicMock()
    mock_response.results = [
        MagicMock(
            title="Weather in San Francisco",
            url="https://weather.com/sf",
            snippet="Current weather: 65°F, partly cloudy",
        )
    ]
    return mock_response


@pytest.fixture
def websearch_logger():
    """Create a WebSearchInterceptionLogger instance"""
    return WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI, LlmProviders.MINIMAX])


@pytest.mark.asyncio
async def test_websearch_chat_completion_hook_detection():
    """Test that websearch hook correctly detects tool calls in response."""
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        Choices,
        Function,
        Message,
    )

    websearch_logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    # Mock response with litellm_web_search tool call
    mock_response = ModelResponse(
        id="test-123",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=Function(
                                name="litellm_web_search",
                                arguments='{"query": "weather in SF"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        model="gpt-4o",
        object="chat.completion",
        created=1234567890,
    )

    # Test should_run_chat_completion_agentic_loop
    should_run, tools_dict = await websearch_logger.async_should_run_chat_completion_agentic_loop(
        response=mock_response,
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "litellm_web_search"},
            }
        ],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    # Verify hook detected the tool call
    assert should_run is True
    assert "tool_calls" in tools_dict
    assert len(tools_dict["tool_calls"]) == 1
    assert tools_dict["tool_calls"][0]["name"] == "litellm_web_search"
    assert tools_dict["response_format"] == "openai"


@pytest.mark.asyncio
async def test_websearch_not_triggered_without_tool():
    """Test that websearch hook is NOT triggered when no web search tool in request."""
    from litellm.types.utils import Choices, Message

    websearch_logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    mock_response = ModelResponse(
        id="test-123",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    role="assistant",
                    content="Here's the answer",
                    tool_calls=None,
                ),
            )
        ],
        model="gpt-4o",
        object="chat.completion",
        created=1234567890,
    )

    # Test without web search tool
    should_run, tools_dict = await websearch_logger.async_should_run_chat_completion_agentic_loop(
        response=mock_response,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "some_other_tool"},
            }
        ],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    # Verify hook did NOT trigger
    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_websearch_not_triggered_for_disabled_provider():
    """Test that websearch hook is NOT triggered for providers not in enabled_providers."""
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        Choices,
        Function,
        Message,
    )

    # Only enable bedrock
    websearch_logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK])

    mock_response = ModelResponse(
        id="test-123",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=Function(
                                name="litellm_web_search",
                                arguments='{"query": "test"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        model="gpt-4o",
        object="chat.completion",
        created=1234567890,
    )

    # Test with OpenAI provider (not enabled)
    should_run, tools_dict = await websearch_logger.async_should_run_chat_completion_agentic_loop(
        response=mock_response,
        model="gpt-4o",
        messages=[{"role": "user", "content": "test"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "litellm_web_search"},
            }
        ],
        stream=False,
        custom_llm_provider="openai",  # Not in enabled_providers
        kwargs={},
    )

    # Verify hook did NOT trigger
    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_websearch_json_serialization_fix():
    """Test that tool call arguments are properly JSON serialized.

    Regression test for the bug where arguments were converted to Python
    string representation instead of proper JSON, causing providers like
    MiniMax to reject requests with 'invalid function arguments json string'.
    """
    from litellm.integrations.websearch_interception.transformation import (
        WebSearchTransformation,
    )

    # Mock tool calls with dict input
    tool_calls = [
        {
            "id": "call_123",
            "name": "litellm_web_search",
            "input": {"query": "weather in SF"},  # Dict input
        }
    ]

    search_results = ["Weather: 65°F, partly cloudy"]

    # Transform to OpenAI format
    assistant_message, tool_messages = WebSearchTransformation.transform_response(
        tool_calls=tool_calls,
        search_results=search_results,
        response_format="openai",
    )

    # Verify arguments are properly JSON serialized
    import json

    arguments_str = assistant_message["tool_calls"][0]["function"]["arguments"]

    # Should be valid JSON
    parsed_args = json.loads(arguments_str)
    assert parsed_args == {"query": "weather in SF"}

    # Should NOT be Python string representation like "{'query': 'weather in SF'}"
    assert arguments_str == '{"query": "weather in SF"}'
    assert arguments_str != "{'query': 'weather in SF'}"


@pytest.mark.asyncio
async def test_maybe_run_chat_completion_agentic_loop_calls_chat_completion_hook():
    """Regression test: maybe_run_chat_completion_agentic_loop must call
    async_should_run_chat_completion_agentic_loop, not async_should_run_agentic_loop.

    Before the fix, the function used the wrong gate check and wrong hook,
    causing WebSearchInterceptionLogger to never intercept chat completion requests
    even when the LLM returned a litellm_web_search tool call.
    """
    from litellm.litellm_core_utils.chat_completion_agentic_loop import (
        maybe_run_chat_completion_agentic_loop,
    )
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        Choices,
        Function,
        Message,
    )

    mock_response = ModelResponse(
        id="test-regression-123",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_abc",
                            type="function",
                            function=Function(
                                name="litellm_web_search",
                                arguments='{"query": "latest news"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        model="gpt-4o",
        object="chat.completion",
        created=1234567890,
    )

    sentinel = ModelResponse(
        id="sentinel-final",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="Here is the news."),
            )
        ],
        model="gpt-4o",
        object="chat.completion",
        created=1234567890,
    )

    websearch_logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    chat_completion_hook_called = False

    async def fake_should_run_chat_completion(response, model, messages, tools, stream, custom_llm_provider, kwargs):
        nonlocal chat_completion_hook_called
        chat_completion_hook_called = True
        return True, {
            "tool_calls": [{"id": "call_abc", "name": "litellm_web_search", "input": {"query": "latest news"}}],
            "tool_type": "websearch",
            "provider": "openai",
            "response_format": "openai",
        }

    async def fake_build_plan(tools, model, messages, response, optional_params, logging_obj, stream, kwargs):
        from litellm.types.integrations.custom_logger import AgenticLoopPlan

        return AgenticLoopPlan(run_agentic_loop=False, response_override=sentinel)

    websearch_logger.async_should_run_chat_completion_agentic_loop = fake_should_run_chat_completion
    websearch_logger.async_build_chat_completion_agentic_loop_plan = fake_build_plan

    import litellm as _litellm

    original_callbacks = _litellm.callbacks[:]
    _litellm.callbacks = [websearch_logger]

    mock_logging_obj = MagicMock()
    mock_logging_obj.dynamic_success_callbacks = None

    try:
        result = await maybe_run_chat_completion_agentic_loop(
            response=mock_response,
            model="gpt-4o",
            messages=[{"role": "user", "content": "Latest news?"}],
            optional_params={
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "litellm_web_search"},
                    }
                ]
            },
            kwargs={},
            logging_obj=mock_logging_obj,
            custom_llm_provider="openai",
            stream=False,
        )
    finally:
        _litellm.callbacks = original_callbacks

    assert chat_completion_hook_called, (
        "async_should_run_chat_completion_agentic_loop was never called; "
        "maybe_run_chat_completion_agentic_loop used the wrong hook"
    )
    assert result is sentinel, "Expected agentic loop to return sentinel final response"


@pytest.mark.asyncio
async def test_execute_chat_completion_agentic_loop_strips_tool_choice():
    """Regression: _execute_chat_completion_agentic_loop must not forward tool_choice
    from the original request into the follow-up synthesis call.

    When the original request forces tool_choice to litellm_web_search, merging
    optional_params into the follow-up params without explicit removal causes the
    model to call the search tool again instead of synthesizing an answer.
    """
    from unittest.mock import patch

    websearch_logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.OPENAI])

    captured_kwargs: dict = {}

    async def fake_acompletion(**kwargs):
        captured_kwargs.update(kwargs)
        return ModelResponse(id="followup", model="gpt-4o", object="chat.completion")

    async def fake_search(query):
        return ("Bitcoin price is $60,000", None)

    with patch.object(websearch_logger, "_execute_search", side_effect=fake_search):
        with patch("litellm.acompletion", side_effect=fake_acompletion):
            await websearch_logger._execute_chat_completion_agentic_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "What is Bitcoin price?"}],
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "litellm_web_search",
                        "input": {"query": "bitcoin price"},
                    }
                ],
                optional_params={
                    "tools": [{"type": "function", "function": {"name": "litellm_web_search"}}],
                    "tool_choice": {"type": "function", "function": {"name": "litellm_web_search"}},
                    "max_tokens": 512,
                },
                logging_obj=MagicMock(),
                stream=False,
                kwargs={},
            )

    assert "tool_choice" not in captured_kwargs, (
        "tool_choice must not appear in follow-up acompletion kwargs; "
        "it would force the model to call the search tool again instead of synthesizing"
    )


if __name__ == "__main__":
    # Run with: pytest test_websearch_chat_completion.py -v -s
    pytest.main([__file__, "-v", "-s"])
