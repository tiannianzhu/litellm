"""
Tests for the Responses API streaming bridge in
litellm/responses/litellm_completion_transformation/streaming_iterator.py.

Ensures that when the underlying chat-completions stream includes tool_calls deltas,
LiteLLM emits Responses API streaming events (output_item.added + function_call_arguments.*).

Also ensures that tool calls that only appear in the final built response still get emitted
before response.completed, and that every event of a bridged stream carries the response id
spend tracking stores, so a follow-up previous_response_id still finds the conversation.
"""

import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.responses.litellm_completion_transformation.custom_tools import native_responses_custom_tool_name_map
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    BaseLiteLLMOpenAIResponseObject,
    ResponsesAPIStreamEvents,
)
from litellm.types.responses.main import build_web_search_call
from litellm.types.utils import (
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

CHAT_COMPLETION_ID = "chatcmpl-77d33d09-effa-4cd2-9c0d-c742d4358256"
RESPONSE_ID_EVENT_TYPES = frozenset(
    {"response.created", "response.in_progress", "response.completed"}
)
_HOSTED_EXEC_GRAMMAR: Final = "\n".join(
    (
        "start: pragma_source | plain_source",
        "pragma_source: PRAGMA_LINE NEWLINE SOURCE",
        "plain_source: SOURCE",
        r"PRAGMA_LINE: /[ \t]*\/\/ @exec:[^\r\n]*/",
        r"NEWLINE: /\r?\n/",
        r"SOURCE: /[\s\S]+/",
    )
)


def _chunk(content: str, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id=CHAT_COMPLETION_ID,
        created=1748575031,
        model="claude-haiku-4-5",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
    )


class _FakeStreamWrapper:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.logging_obj = MagicMock()

    def __iter__(self):
        return self

    def __next__(self):
        if not self._chunks:
            raise StopIteration
        chunk = self._chunks.pop(0)
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        chunk = self._chunks.pop(0)
        if isinstance(chunk, Exception):
            raise chunk
        return chunk


def _build_iterator(
    chunks,
    custom_llm_provider: str = "anthropic",
    responses_api_request=None,
) -> LiteLLMCompletionStreamingIterator:
    return LiteLLMCompletionStreamingIterator(
        model="claude-haiku-4-5",
        litellm_custom_stream_wrapper=_FakeStreamWrapper(chunks),
        request_input="What is the weather in San Francisco?",
        responses_api_request=responses_api_request or {},
        custom_llm_provider=custom_llm_provider,
        litellm_metadata={},
    )


def _response_ids(events) -> list[str]:
    return [
        event.response.id
        for event in events
        if getattr(event, "type", None) in RESPONSE_ID_EVENT_TYPES
    ]


def test_tool_call_delta_is_emitted_as_responses_events():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    # A streaming chunk with tool_calls delta but no text
    chunk = ModelResponseStream(
        id="chunk-1",
        created=123,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "do_thing", "arguments": '{"x":1}'},
                        }
                    ],
                ),
            )
        ],
    )

    evt1 = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)
    assert evt1 is not None
    assert evt1.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    assert evt1.output_index == 1

    # The arguments are now chunked, so we get the first delta chunk
    evt2 = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)
    assert evt2 is not None
    assert evt2.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA
    assert evt2.item_id == "fc_call_1"
    assert evt2.output_index == 1
    # The delta will be a chunk of the arguments, not the full arguments
    assert len(evt2.delta) <= 10  # Chunks are max 10 characters


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize(
    "tool_type,result_kind,expected_sources",
    [
        (
            "web_search",
            "valid",
            {"srvtoolu_01Search": ["https://example.com/one"], "srvtoolu_02Search": ["https://example.com/two"]},
        ),
        (
            "web_search_preview",
            "valid",
            {"srvtoolu_01Search": ["https://example.com/one"], "srvtoolu_02Search": ["https://example.com/two"]},
        ),
        ("function", "valid", {}),
        ("web_search", "unpaired", {"srvtoolu_01Search": ["https://example.com/one"]}),
        ("web_search", "web_fetch", {"srvtoolu_02Search": ["https://example.com/two"]}),
        ("web_search", "error", {"srvtoolu_01Search": [], "srvtoolu_02Search": ["https://example.com/two"]}),
    ],
)
async def test_web_search_stream_preserves_hosted_and_client_calls(sync_mode, tool_type, result_kind, expected_sources):
    call_ids: Final = ("srvtoolu_01Search", "srvtoolu_02Search")
    valid_results: Final = (
        {
            "type": "web_search_tool_result",
            "tool_use_id": call_ids[0],
            "content": [{"type": "web_search_result", "url": "https://example.com/one"}],
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": call_ids[1],
            "content": [{"type": "web_search_result", "url": "https://example.com/two"}],
        },
    )
    first_result: Final = (
        {**valid_results[0], "type": "web_fetch_tool_result"}
        if result_kind == "web_fetch"
        else {**valid_results[0], "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"}}
        if result_kind == "error"
        else valid_results[0]
    )
    results: Final = [first_result] if result_kind == "unpaired" else [first_result, valid_results[1]]
    deltas: Final = (
        Delta(
            role="assistant",
            content=None,
            tool_calls=[
                {"index": 0, "id": call_ids[0], "type": "function", "function": {"name": "web_search", "arguments": ""}}
            ],
            provider_specific_fields={
                "web_search_calls": [
                    build_web_search_call(
                        call_ids[0],
                        {},
                        {"content": []},
                        status="in_progress",
                    )
                ]
                if tool_type != "function" and result_kind != "web_fetch"
                else [],
            },
        ),
        Delta(
            content=None,
            tool_calls=[
                {
                    "index": 1,
                    "id": "toolu_regular",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        ),
        Delta(content=None, tool_calls=[{"index": 0, "function": {"arguments": '{"query":'}}]),
        Delta(content=None, tool_calls=[{"index": 0, "function": {"arguments": '"one"}'}}]),
        Delta(
            content=None,
            provider_specific_fields={
                "web_search_results": [first_result],
                "web_search_calls": [
                    build_web_search_call(call_ids[0], {"query": "one"}, first_result)
                ]
                if tool_type != "function" and first_result["type"] == "web_search_tool_result"
                else [],
            },
        ),
        Delta(
            content=None,
            provider_specific_fields={
                "web_search_results": results,
                "web_search_calls": [
                    build_web_search_call(
                        result["tool_use_id"],
                        {"query": "one" if result["tool_use_id"].endswith("01Search") else "two"},
                        result,
                    )
                    for result in results
                    if tool_type != "function" and result["type"] == "web_search_tool_result"
                ],
            },
        ),
        Delta(
            content="answer",
            tool_calls=[
                {
                    "index": 2,
                    "id": call_ids[1],
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"two"}'},
                }
            ],
        ),
    )
    chunks: Final = tuple(
        ModelResponseStream(
            id=CHAT_COMPLETION_ID,
            created=1748575031,
            model="claude-fable-5-1",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(index=0, delta=delta, finish_reason="stop" if index == len(deltas) - 1 else None)
            ],
        )
        for index, delta in enumerate(deltas)
    )
    request_tools: Final = (
        [{"type": "function", "name": "web_search", "parameters": {"type": "object"}}]
        if tool_type == "function"
        else [{"type": tool_type}]
    )
    iterator: Final = LiteLLMCompletionStreamingIterator(
        model="claude-fable-5-1",
        litellm_custom_stream_wrapper=_FakeStreamWrapper(chunks),
        request_input="search",
        responses_api_request={"tools": request_tools},
        custom_llm_provider="anthropic",
    )
    events: Final = (
        [event.model_dump(exclude_none=True) for event in iterator]
        if sync_mode
        else [event.model_dump(exclude_none=True) async for event in iterator]
    )
    completed: Final = events[-1]
    search_items: Final = {
        item["id"].removeprefix("ws_"): item
        for item in completed["response"]["output"]
        if item["type"] == "web_search_call"
    }
    function_items: Final = {
        item["call_id"]: item for item in completed["response"]["output"] if item["type"] == "function_call"
    }
    function_events: Final = [event for event in events if "function_call_arguments" in event["type"]]
    expected_functions: Final = set(call_ids).difference(expected_sources) | {"toolu_regular"}
    search_indexes: Final = {
        event["output_index"] for event in events if event["type"] == "response.web_search_call.completed"
    }
    completed_indexes: Final = {item["id"]: index for index, item in enumerate(completed["response"]["output"])}

    assert completed["type"] == "response.completed"
    assert [item["content"][0]["text"] for item in completed["response"]["output"] if item["type"] == "message"] == [
        "answer"
    ]
    assert set(search_items) == set(expected_sources)
    assert set(function_items) == expected_functions
    assert {event["item_id"] for event in function_events} == {item["id"] for item in function_items.values()}
    assert len(search_indexes) == len(expected_sources)
    for call_id, item in search_items.items():
        search_events = [
            event for event in events if event.get("item_id", event.get("item", {}).get("id")) == item["id"]
        ]
        assert [event["type"] for event in search_events] == [
            "response.output_item.added",
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
            "response.output_item.done",
        ]
        assert {event["output_index"] for event in search_events} == {completed_indexes[item["id"]]}
        assert search_events[0]["item"]["status"] == "in_progress"
        assert search_events[-1]["item"] == item
        assert item["status"] == (
            "failed" if result_kind == "error" and call_id.endswith("01Search") else "completed"
        )
        assert item["action"]["type"] == "search"
        assert item["action"]["query"] == ("one" if call_id.endswith("01Search") else "two")
        assert item["action"]["queries"] == [item["action"]["query"]]
        assert [source["url"] for source in item["action"]["sources"]] == expected_sources[call_id]
    for call_id, item in function_items.items():
        argument_deltas = [
            event["delta"]
            for event in function_events
            if event["item_id"] == item["id"] and event["type"].endswith(".delta")
        ]
        assert json.loads("".join(argument_deltas)) == json.loads(item["arguments"])
        assert json.loads(item["arguments"]) == (
            {"city": "Paris"}
            if call_id == "toolu_regular"
            else {"query": "one" if call_id.endswith("01Search") else "two"}
        )
        assert any(
            event["type"] == "response.output_item.done"
            and event.get("item") == item
            and event["output_index"] == completed_indexes[item["id"]]
            for event in events
        )


def test_tool_calls_present_only_in_final_response_are_emitted_before_completed():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    # Construct a final ModelResponse with tool_calls on the message.
    # We bypass the stream builder and directly set iterator.litellm_model_response.
    response = ModelResponse(
        id="resp-1",
        created=123,
        model="test-model",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "do_thing", "arguments": '{"y":2}'},
                            "index": 0,
                        }
                    ],
                },
            }
        ],
    )
    iterator.litellm_model_response = response

    # First common_done_event_logic call should yield tool events, not response.completed.
    evt1 = iterator.common_done_event_logic(sync_mode=True)
    assert evt1.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    assert evt1.output_index == 1

    # Now delta events are emitted (arguments split into chunks)
    # Collect all delta events
    delta_events = []
    while True:
        evt = iterator.common_done_event_logic(sync_mode=True)
        if evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA:
            delta_events.append(evt)
        else:
            break

    # Verify we got delta events
    assert len(delta_events) > 0
    # Verify they reconstruct the original arguments
    concatenated_args = "".join(evt.delta for evt in delta_events)
    assert concatenated_args == '{"y":2}'

    # The last event should be FUNCTION_CALL_ARGUMENTS_DONE
    assert evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE
    assert evt.item_id == "fc_call_2"
    assert evt.output_index == 1
    assert evt.arguments == '{"y":2}'

    evt_final = iterator.common_done_event_logic(sync_mode=True)
    assert evt_final.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
    assert evt_final.output_index == 1


def test_tool_call_arguments_are_chunked_to_match_openai_behavior():
    """
    Test that large tool call arguments are split into smaller chunks (size 10)
    to replicate OpenAI's native streaming behavior.

    This is especially important for providers like Bedrock that send complete
    arguments at once, which need to be split to match OpenAI's token-by-token streaming.
    """
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    # Create a chunk with a large arguments string that should be split
    large_arguments = (
        '{"param1": "value1", "param2": "value2", "param3": "value3"}'  # 67 chars
    )
    chunk = ModelResponseStream(
        id="chunk-1",
        created=123,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_test",
                            "type": "function",
                            "function": {
                                "name": "test_function",
                                "arguments": large_arguments,
                            },
                        }
                    ],
                ),
            )
        ],
    )

    # Process the chunk once - it queues all events internally
    evt = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)

    # First event should be OUTPUT_ITEM_ADDED
    assert evt is not None
    assert evt.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    assert evt.output_index == 1
    assert hasattr(evt, "__dict__") and "sequence_number" in evt.__dict__

    # Collect all remaining delta events from the pending queue by creating empty chunks
    delta_events = []
    empty_chunk = ModelResponseStream(
        id="chunk-1",
        created=123,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(role="assistant", content=""),
            )
        ],
    )

    # Keep draining pending events (expected: ceil(67 / 10) = 7 delta events)
    while iterator._pending_tool_events:
        evt = iterator._transform_chat_completion_chunk_to_response_api_chunk(
            empty_chunk
        )
        if evt and evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA:
            delta_events.append(evt)

    # Verify multiple delta events were created (at least 6 chunks for 67 chars)
    assert len(delta_events) >= 6  # 67 chars split into chunks of max 10 chars each

    # Verify each delta is at most 10 characters
    for evt in delta_events:
        assert len(evt.delta) <= 10
        assert evt.item_id == "fc_call_test"
        assert evt.output_index == 1
        assert hasattr(evt, "__dict__") and "sequence_number" in evt.__dict__

    # Verify all deltas concatenated equal the original arguments
    concatenated = "".join(evt.delta for evt in delta_events)
    assert concatenated == large_arguments

    # Verify sequence numbers are increasing
    sequence_numbers = [evt.__dict__["sequence_number"] for evt in delta_events]
    assert sequence_numbers == sorted(sequence_numbers)
    assert len(set(sequence_numbers)) == len(sequence_numbers)  # All unique


def test_tool_call_delta_without_id_uses_index_mapping():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    chunks = [
        [
            {
                "index": 0,
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"lo'},
            }
        ],
        [{"index": 0, "type": "function", "function": {"arguments": 'cation":'}}],
        [{"index": 0, "type": "function", "function": {"arguments": ' "New'}}],
        [{"index": 0, "type": "function", "function": {"arguments": ' York"}'}}],
    ]

    for tool_calls in chunks:
        iterator._queue_tool_call_delta_events(tool_calls)

    all_events = []
    while iterator._pending_tool_events:
        all_events.append(iterator._pending_tool_events.pop(0))

    delta_events = [
        evt
        for evt in all_events
        if evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA
    ]
    streamed_arguments = "".join(evt.delta for evt in delta_events)

    assert streamed_arguments == '{"location": "New York"}'

    output_item_added_events = [
        evt
        for evt in all_events
        if evt.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    ]
    assert len(output_item_added_events) == 1
    assert output_item_added_events[0].item.id == "fc_call_abc123"
    assert output_item_added_events[0].item.call_id == "call_abc123"


def test_parallel_tool_calls_without_ids_use_index_mapping():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    iterator._queue_tool_call_delta_events(
        [
            {
                "index": 0,
                "id": "call_a",
                "type": "function",
                "function": {"name": "tool_a", "arguments": '{"x":'},
            },
            {
                "index": 1,
                "id": "call_b",
                "type": "function",
                "function": {"name": "tool_b", "arguments": '{"y":'},
            },
        ]
    )
    iterator._queue_tool_call_delta_events(
        [
            {"index": 0, "type": "function", "function": {"arguments": "1}"}},
            {"index": 1, "type": "function", "function": {"arguments": "2}"}},
        ]
    )

    all_events = []
    while iterator._pending_tool_events:
        all_events.append(iterator._pending_tool_events.pop(0))

    output_item_added_events = [
        evt
        for evt in all_events
        if evt.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    ]
    assert len(output_item_added_events) == 2

    delta_events = [
        evt
        for evt in all_events
        if evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA
    ]
    arguments_by_call_id = {}
    for evt in delta_events:
        arguments_by_call_id.setdefault(evt.item_id, "")
        arguments_by_call_id[evt.item_id] += evt.delta

    assert arguments_by_call_id["fc_call_a"] == '{"x":1}'
    assert arguments_by_call_id["fc_call_b"] == '{"y":2}'


def test_reused_index_with_new_call_id_marks_fallback_ambiguous():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    iterator._queue_tool_call_delta_events(
        [
            {
                "index": 0,
                "id": "call_a",
                "type": "function",
                "function": {"name": "tool_a", "arguments": '{"a":'},
            }
        ]
    )
    iterator._queue_tool_call_delta_events(
        [
            {
                "index": 0,
                "id": "call_b",
                "type": "function",
                "function": {"name": "tool_b", "arguments": '{"b":'},
            }
        ]
    )
    # Ambiguous chunk: index reused and id missing. We should skip fallback rather than misroute.
    iterator._queue_tool_call_delta_events(
        [
            {
                "index": 0,
                "type": "function",
                "function": {"arguments": "1}"},
            }
        ]
    )

    all_events = []
    while iterator._pending_tool_events:
        all_events.append(iterator._pending_tool_events.pop(0))

    delta_events = [
        evt
        for evt in all_events
        if evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA
    ]
    arguments_by_call_id = {}
    for evt in delta_events:
        arguments_by_call_id.setdefault(evt.item_id, "")
        arguments_by_call_id[evt.item_id] += evt.delta

    assert arguments_by_call_id["fc_call_a"] == '{"a":'
    assert arguments_by_call_id["fc_call_b"] == '{"b":'
    assert arguments_by_call_id["fc_call_a"] != '{"a":1}'
    assert arguments_by_call_id["fc_call_b"] != '{"b":1}'


@pytest.mark.asyncio
async def test_streaming_events_share_the_chat_completion_response_id():
    """
    Every event of a bridged stream has to carry the same id, and that id has to decode
    to the chat completion id spend tracking stores as `request_id`. Otherwise a
    follow-up `previous_response_id` matches no session and the conversation is dropped.
    """
    iterator = _build_iterator([_chunk("Hello"), _chunk("!", finish_reason="stop")])

    events = [event async for event in iterator]

    response_ids = _response_ids(events)
    assert len(response_ids) == 3
    assert len(set(response_ids)) == 1
    decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(response_ids[0])
    assert decoded["response_id"] == CHAT_COMPLETION_ID
    assert decoded["custom_llm_provider"] == "anthropic"


def test_sync_streaming_events_share_the_chat_completion_response_id():
    iterator = _build_iterator([_chunk("Hello"), _chunk("!", finish_reason="stop")])

    events = list(iterator)

    response_ids = _response_ids(events)
    assert len(response_ids) == 3
    assert len(set(response_ids)) == 1
    assert (
        ResponsesAPIRequestUtils._decode_responses_api_response_id(response_ids[0])["response_id"]
        == CHAT_COMPLETION_ID
    )


@pytest.mark.asyncio
async def test_streaming_emits_every_chunk_after_priming_the_response_id():
    iterator = _build_iterator(
        [_chunk("Hel"), _chunk("lo"), _chunk("!", finish_reason="stop")]
    )

    events = [event async for event in iterator]

    deltas = "".join(
        event.delta for event in events if getattr(event, "type", None) == "response.output_text.delta"
    )
    assert deltas == "Hello!"


@pytest.mark.asyncio
async def test_streaming_response_id_falls_back_when_upstream_yields_nothing():
    iterator = _build_iterator([])

    events = [event async for event in iterator]

    response_ids = _response_ids(events)
    assert response_ids
    assert len(set(response_ids)) == 1
    assert response_ids[0].startswith("resp_")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_hosted_stream_without_finish_reason_emits_failed_event(sync_mode: bool):
    iterator = _build_iterator([_chunk("partial")], custom_llm_provider="hosted_vllm")

    events = await _collect_events(iterator, sync_mode)
    event_types = tuple(event.type for event in events)
    failed_event = next(event for event in events if event.type == ResponsesAPIStreamEvents.RESPONSE_FAILED)

    assert ResponsesAPIStreamEvents.RESPONSE_COMPLETED not in event_types
    assert failed_event.response.status == "failed"
    assert failed_event.response.error == {
        "code": "server_error",
        "message": "upstream stream ended without a finish reason",
    }


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_hosted_stream_transport_error_is_reraised(sync_mode: bool):
    iterator = _build_iterator(
        [_chunk("partial"), RuntimeError("transport closed")],
        custom_llm_provider="hosted_vllm",
    )

    with pytest.raises(RuntimeError, match="transport closed"):
        await _collect_events(iterator, sync_mode)

    assert iterator.finished is True


def test_completed_event_restores_usage_hidden_by_stream_options_none():
    final_chunk = _chunk("", finish_reason="stop")
    final_chunk._hidden_params = {"usage": Usage(prompt_tokens=117, completion_tokens=5, total_tokens=122)}
    iterator = _build_iterator([_chunk("the document says hello"), final_chunk])

    events = list(iterator)

    completed = next(
        event for event in events if getattr(event, "type", None) == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    )
    assert completed.response.usage.input_tokens == 117
    assert completed.response.usage.output_tokens == 5


def _empty_choices_chunk(usage: Usage | None = None) -> ModelResponseStream:
    return ModelResponseStream(id=CHAT_COMPLETION_ID, model="claude-haiku-4-5", choices=[], usage=usage)


@pytest.mark.asyncio
async def test_leading_empty_choices_chunk_does_not_kill_the_stream():
    """
    Azure leads some streams with a `prompt_filter_results` chunk whose `choices` is empty.
    The bridge used to index `choices[0]` on it and die before the first token.
    """
    iterator = _build_iterator([_empty_choices_chunk(), _chunk("Hello"), _chunk("!", finish_reason="stop")])

    events = [event async for event in iterator]

    event_types = [getattr(event, "type", None) for event in events]
    assert event_types.count(ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED) == 1
    assert "".join(event.delta for event in events if event.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA) == "Hello!"
    assert event_types[-1] == ResponsesAPIStreamEvents.RESPONSE_COMPLETED


@pytest.mark.asyncio
async def test_trailing_empty_choices_usage_chunk_reaches_response_completed():
    """
    With `stream_options.include_usage` (which the bridge always sets) the last upstream chunk
    carries only usage and an empty `choices`. It must not crash the stream, and its usage must
    still land on `response.completed`.
    """
    usage: Final = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    iterator = _build_iterator([_chunk("Hello"), _chunk("", finish_reason="stop"), _empty_choices_chunk(usage)])

    events = [event async for event in iterator]

    completed = next(
        event for event in events if getattr(event, "type", None) == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    )
    assert completed.response.usage.input_tokens == 10
    assert completed.response.usage.output_tokens == 5


def test_is_reasoning_end_ignores_empty_choices_chunk():
    assert _build_iterator([])._is_reasoning_end(_empty_choices_chunk()) is False


def test_object_tool_call_arguments_stream_as_valid_json():
    """A provider that sends decoded object arguments must still stream valid JSON.

    `str()` on a dict yields a Python repr with single quotes, which clients
    parsing function_call_arguments reject with errors like
    "Expecting ',' delimiter".
    """
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )
    iterator._queue_tool_call_delta_events(
        [
            {
                "index": 0,
                "id": "call_obj",
                "type": "function",
                "function": {"name": "shell", "arguments": {"command": "ls", "flags": ["-l"]}},
            }
        ]
    )

    streamed_arguments = "".join(
        evt.delta
        for evt in iterator._pending_tool_events
        if evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA
    )

    assert json.loads(streamed_arguments) == {"command": "ls", "flags": ["-l"]}


def test_streamed_anthropic_tool_call_events_correlate_on_normalized_item_id():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    response = ModelResponse(
        id="resp-anthropic",
        created=123,
        model="test-model",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "toolu_01AbCdEf",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            "index": 0,
                        }
                    ],
                },
            }
        ],
    )
    iterator.litellm_model_response = response

    events = []
    while True:
        evt = iterator.common_done_event_logic(sync_mode=True)
        events.append(evt)
        if evt.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE:
            break

    added = [e for e in events if e.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED]
    deltas = [e for e in events if e.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA]
    dones = [e for e in events if e.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE]
    item_dones = [e for e in events if e.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE]

    assert len(added) == 1 and len(dones) == 1 and len(item_dones) == 1 and deltas
    assert added[0].item.id == "fc_toolu_01AbCdEf"
    assert added[0].item.call_id == "toolu_01AbCdEf"
    assert item_dones[0].item.id == "fc_toolu_01AbCdEf"
    assert item_dones[0].item.call_id == "toolu_01AbCdEf"
    for evt in deltas + dones:
        assert evt.item_id == added[0].item.id


def _tool_call_chunk(finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id=CHAT_COMPLETION_ID,
        created=1748575031,
        model="claude-haiku-4-5",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_pwd",
                            "type": "function",
                            "function": {"name": "run_command", "arguments": '{"command":"pwd"}'},
                            "index": 0,
                        }
                    ],
                ),
                finish_reason=finish_reason,
            )
        ],
    )


def _custom_tool_call_chunk(
    wire_name: str,
    arguments: str,
    call_id: str | None,
    finish_reason: str | None = None,
) -> ModelResponseStream:
    tool_call = {
        "index": 0,
        "type": "function",
        "function": {"name": wire_name, "arguments": arguments},
    }
    if call_id is not None:
        tool_call["id"] = call_id
    return ModelResponseStream(
        id=CHAT_COMPLETION_ID,
        created=1748575031,
        model="claude-haiku-4-5",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(role="assistant", content=None, tool_calls=[tool_call]),
                finish_reason=finish_reason,
            )
        ],
    )


def test_streamed_named_tool_choice_is_echoed_in_responses_api_shape() -> None:
    iterator: Final = LiteLLMCompletionStreamingIterator(
        model="claude-haiku-4-5",
        litellm_custom_stream_wrapper=_FakeStreamWrapper([_tool_call_chunk(finish_reason="tool_calls")]),
        request_input="Run the command pwd.",
        responses_api_request={
            "tools": [{"type": "function", "name": "run_command", "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "run_command"},
        },
        custom_llm_provider="anthropic",
        litellm_metadata={},
    )

    events: Final = list(iterator)

    response_events: Final = [event for event in events if getattr(event, "type", None) in RESPONSE_ID_EVENT_TYPES]
    assert [event.type for event in response_events] == [
        "response.created",
        "response.in_progress",
        "response.completed",
    ]
    assert [event.response.tool_choice for event in response_events] == [
        {"type": "function", "name": "run_command"},
        {"type": "function", "name": "run_command"},
        {"type": "function", "name": "run_command"},
    ]
    assert any(getattr(event, "type", None) == "response.output_item.done" for event in events)


def test_streamed_unrecognized_tool_choice_is_echoed_as_auto() -> None:
    iterator: Final = LiteLLMCompletionStreamingIterator(
        model="claude-haiku-4-5",
        litellm_custom_stream_wrapper=_FakeStreamWrapper([_tool_call_chunk(finish_reason="tool_calls")]),
        request_input="Run the command pwd.",
        responses_api_request={
            "tools": [{"type": "function", "name": "run_command", "parameters": {"type": "object"}}],
            "tool_choice": "any",
        },
        custom_llm_provider="anthropic",
        litellm_metadata={},
    )

    response_events: Final = [
        event for event in iterator if getattr(event, "type", None) in RESPONSE_ID_EVENT_TYPES
    ]

    assert [event.response.tool_choice for event in response_events] == ["auto", "auto", "auto"]


def _reasoning_chunk(reasoning: str, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id=CHAT_COMPLETION_ID,
        created=1748575031,
        model="claude-haiku-4-5",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(role="assistant", reasoning_content=reasoning),
                finish_reason=finish_reason,
            )
        ],
    )


def _role_only_chunk() -> ModelResponseStream:
    return ModelResponseStream(
        id=CHAT_COMPLETION_ID,
        created=1748575031,
        model="claude-haiku-4-5",
        object="chat.completion.chunk",
        choices=[StreamingChoices(index=0, delta=Delta(role="assistant"), finish_reason=None)],
    )


async def _collect_events(
    iterator: LiteLLMCompletionStreamingIterator, sync_mode: bool
) -> list[BaseLiteLLMOpenAIResponseObject]:
    if sync_mode:
        return list(iterator)
    return [event async for event in iterator]


async def _collect_events_until_bad_gateway(
    iterator: LiteLLMCompletionStreamingIterator, sync_mode: bool
) -> tuple[tuple[BaseLiteLLMOpenAIResponseObject, ...], litellm.BadGatewayError]:
    events = []
    while True:
        try:
            event = next(iterator) if sync_mode else await iterator.__anext__()
        except litellm.BadGatewayError as exc:
            return tuple(events), exc
        events.append(event)


def _is_message_item(event: BaseLiteLLMOpenAIResponseObject) -> bool:
    return getattr(getattr(event, "item", None), "type", None) == "message"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_tool_only_stream_emits_no_message_item_events(sync_mode: bool):
    iterator: Final = _build_iterator([_tool_call_chunk(), _chunk("", finish_reason="tool_calls")])

    events: Final = await _collect_events(iterator, sync_mode)

    message_item_events = [
        event
        for event in events
        if getattr(event, "type", None)
        in (ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE)
        and _is_message_item(event)
    ]
    assert message_item_events == []
    assert [
        event
        for event in events
        if str(getattr(event, "type", "")).startswith("response.output_text")
        or getattr(event, "type", None)
        in (ResponsesAPIStreamEvents.CONTENT_PART_ADDED, ResponsesAPIStreamEvents.CONTENT_PART_DONE)
    ] == []
    assert any(getattr(event, "type", None) == ResponsesAPIStreamEvents.RESPONSE_COMPLETED for event in events)


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_reasoning_then_text_announces_message_item_before_text_events(sync_mode: bool):
    iterator: Final = _build_iterator(
        [
            _reasoning_chunk("let me think"),
            _chunk("Hello"),
            _chunk("!", finish_reason="stop"),
        ]
    )

    events: Final = await _collect_events(iterator, sync_mode)

    announced_message_ids: set[str] = set()
    announced_indexes_by_item_type: dict[str, int] = {}
    content_part_added_seen = False
    saw_text_delta = False
    for event in events:
        event_type = getattr(event, "type", None)
        if event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED:
            announced_indexes_by_item_type[event.item.type] = event.output_index
            if _is_message_item(event):
                announced_message_ids.add(event.item.id)
        elif event_type == ResponsesAPIStreamEvents.CONTENT_PART_ADDED:
            content_part_added_seen = True
        elif event_type in (
            ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
            ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
            ResponsesAPIStreamEvents.CONTENT_PART_DONE,
        ):
            assert event.item_id in announced_message_ids
            if event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA:
                assert content_part_added_seen
                saw_text_delta = True
        elif event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE and _is_message_item(event):
            assert event.item.id in announced_message_ids
    assert saw_text_delta
    assert "".join(
        event.delta for event in events if getattr(event, "type", None) == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
    ) == "Hello!"
    assert announced_indexes_by_item_type["message"] != announced_indexes_by_item_type["reasoning"]


@pytest.mark.asyncio
async def test_reasoning_item_closes_before_message_item_opens():
    iterator: Final = _build_iterator(
        [
            _reasoning_chunk("let me think"),
            _chunk("Hello"),
            _chunk("!", finish_reason="stop"),
        ]
    )

    events: Final = await _collect_events(iterator, sync_mode=False)

    item_lifecycle: Final = [
        (event.type, event.item.type)
        for event in events
        if getattr(event, "type", None)
        in (ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE)
    ]
    assert item_lifecycle == [
        (ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, "reasoning"),
        (ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE, "reasoning"),
        (ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, "message"),
        (ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE, "message"),
    ]


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_role_only_chunk_does_not_preempt_reasoning_item(sync_mode: bool):
    iterator: Final = _build_iterator(
        [
            _role_only_chunk(),
            _reasoning_chunk("first reasoning part "),
            _reasoning_chunk("second reasoning part"),
            _chunk("answer"),
            _tool_call_chunk(finish_reason="tool_calls"),
        ]
    )

    events: Final = await _collect_events(iterator, sync_mode)
    output_item_events: Final = [
        event
        for event in events
        if getattr(event, "type", None)
        in (ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE)
    ]
    reasoning_events: Final = [
        event for event in output_item_events if getattr(event.item, "type", None) == "reasoning"
    ]
    message_events: Final = [event for event in output_item_events if _is_message_item(event)]
    tool_events: Final = [event for event in output_item_events if getattr(event.item, "type", None) == "function_call"]

    assert output_item_events[0].item.type == "reasoning"
    assert [event.type for event in reasoning_events] == [
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
    ]
    assert reasoning_events[0].item.id == reasoning_events[1].item.id
    assert reasoning_events[1].item.content[0].text == "first reasoning part second reasoning part"
    assert [event.type for event in message_events] == [
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
    ]
    assert [event.type for event in tool_events] == [
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
    ]


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_reasoning_done_item_matches_completed_reasoning_content(sync_mode: bool):
    iterator: Final = _build_iterator(
        [
            _reasoning_chunk("first reasoning part "),
            _reasoning_chunk("second reasoning part"),
            _chunk("answer", finish_reason="stop"),
        ]
    )

    events: Final = await _collect_events(iterator, sync_mode)
    reasoning_done: Final = next(
        event
        for event in events
        if getattr(event, "type", None) == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
        and getattr(event.item, "type", None) == "reasoning"
    )
    completed: Final = next(
        event for event in events if getattr(event, "type", None) == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    )
    completed_reasoning: Final = next(item for item in completed.response.output if item.type == "reasoning")
    reasoning_done_item: Final = reasoning_done.model_dump(mode="json", exclude_none=True)["item"]
    completed_reasoning_item: Final = next(
        item
        for item in completed.model_dump(mode="json", exclude_none=True)["response"]["output"]
        if item["type"] == "reasoning"
    )

    assert reasoning_done.item.id == completed_reasoning.id
    assert reasoning_done.item.status == completed_reasoning.status == "completed"
    assert reasoning_done_item == completed_reasoning_item
    assert reasoning_done_item["content"] == [
        {
            "type": "reasoning_text",
            "text": "first reasoning part second reasoning part",
            "annotations": [],
        }
    ]


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.asyncio
async def test_terminal_reasoning_delta_precedes_done_events(sync_mode: bool, finish_reason: str):
    iterator: Final = _build_iterator([_reasoning_chunk("last", finish_reason=finish_reason)])

    events: Final = await _collect_events(iterator, sync_mode)
    reasoning_types: Final = [
        event.type
        for event in events
        if event.type
        in (
            ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
            ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DONE,
            ResponsesAPIStreamEvents.REASONING_SUMMARY_PART_DONE,
        )
        or (event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE and event.item.type == "reasoning")
    ]
    assert reasoning_types == [
        ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
        ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DONE,
        ResponsesAPIStreamEvents.REASONING_SUMMARY_PART_DONE,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
    ]


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_empty_text_response_emits_message_lifecycle(sync_mode: bool):
    iterator: Final = _build_iterator([_role_only_chunk(), _chunk("", finish_reason="stop")])

    events: Final = await _collect_events(iterator, sync_mode)
    assert [event.type for event in events] == [
        ResponsesAPIStreamEvents.RESPONSE_CREATED,
        ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
        ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
        ResponsesAPIStreamEvents.CONTENT_PART_DONE,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
    ]
    added: Final = events[2].model_dump(mode="json", exclude_none=True)
    done: Final = events[-2].model_dump(mode="json", exclude_none=True)
    completed: Final = events[-1].model_dump(mode="json", exclude_none=True)
    assert added["item"]["id"] == done["item"]["id"]
    assert done["item"] == completed["response"]["output"][0]
    assert done["item"]["content"] == [{"type": "output_text", "text": "", "annotations": []}]


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_reasoning_length_termination_is_incomplete(sync_mode: bool):
    iterator: Final = _build_iterator([_reasoning_chunk("unfinished", finish_reason="length")])

    events: Final = await _collect_events(iterator, sync_mode)
    reasoning_done: Final = next(
        event
        for event in events
        if getattr(event, "type", None) == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
        and getattr(event.item, "type", None) == "reasoning"
    )
    incomplete: Final = next(
        event for event in events if getattr(event, "type", None) == ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE
    )
    completed_reasoning: Final = next(item for item in incomplete.response.output if item.type == "reasoning")
    reasoning_done_item: Final = reasoning_done.model_dump(mode="json", exclude_none=True)["item"]
    incomplete_reasoning_item: Final = next(
        item
        for item in incomplete.model_dump(mode="json", exclude_none=True)["response"]["output"]
        if item["type"] == "reasoning"
    )

    assert reasoning_done.item.status == completed_reasoning.status == "incomplete"
    assert reasoning_done_item == incomplete_reasoning_item
    assert reasoning_done_item["content"] == [{"type": "reasoning_text", "text": "unfinished", "annotations": []}]


def test_streaming_custom_wire_names_restore_original_namespaces():
    from litellm.responses.litellm_completion_transformation.custom_tools import native_responses_custom_tool_name_map

    request: Final = {
        "tools": [
            {
                "type": "namespace",
                "name": "shell",
                "tools": [
                    {
                        "type": "custom",
                        "name": "exec",
                        "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
                    }
                ],
            },
            {
                "type": "namespace",
                "name": "filesystem",
                "tools": [
                    {
                        "type": "custom",
                        "name": "exec",
                        "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
                    }
                ],
            },
        ]
    }
    wire_names: Final = native_responses_custom_tool_name_map(request)
    wire_calls: Final = tuple((wire_name, identity) for wire_name, identity in wire_names.items())
    iterator: Final = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request=request,
    )
    tool_calls: Final = [
        {
            "index": index,
            "id": f"call_{index}",
            "function": {"name": wire_name, "arguments": json.dumps({"content": namespace})},
        }
        for index, (wire_name, (_, namespace)) in enumerate(wire_calls)
    ]
    iterator._queue_tool_call_delta_events(tool_calls)
    iterator._queue_final_tool_call_done_events(
        ModelResponse(
            id="chatcmpl-custom",
            created=1,
            model="test-model",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                }
            ],
        )
    )

    items: Final = [
        event.item
        for event in iterator._pending_tool_events
        if event.type in (ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE)
    ]

    assert {(item.call_id, item.name, item.namespace) for item in items} == {
        (f"call_{index}", name, namespace) for index, (_, (name, namespace)) in enumerate(wire_calls)
    }
    assert {item.input for item in items if item.status == "completed"} == {
        namespace for _, (_, namespace) in wire_calls
    }


def test_streaming_hosted_custom_wire_name_does_not_shadow_an_ordinary_function():
    from litellm.responses.litellm_completion_transformation.custom_tools import native_responses_custom_tool_name_map

    request: Final = {
        "tools": [
            {"type": "function", "name": "exec", "parameters": {"type": "object"}},
            {
                "type": "namespace",
                "name": "shell",
                "tools": [
                    {
                        "type": "custom",
                        "name": "exec",
                        "format": {"type": "grammar", "syntax": "lark", "definition": _HOSTED_EXEC_GRAMMAR},
                    }
                ],
            },
        ]
    }
    custom_wire_name: Final = next(iter(native_responses_custom_tool_name_map(request)))
    iterator: Final = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request=request,
        custom_llm_provider="hosted_vllm",
    )
    tool_calls: Final = [
        {"index": 0, "id": "call_function", "function": {"name": "exec", "arguments": '{"path":"/tmp"}'}},
        {
            "index": 1,
            "id": "call_custom",
            "function": {"name": custom_wire_name, "arguments": '{"content":"pwd"}'},
        },
    ]
    iterator._queue_tool_call_delta_events(tool_calls)
    iterator._queue_final_tool_call_done_events(
        ModelResponse(
            id="chatcmpl-custom",
            created=1,
            model="test-model",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                }
            ],
        )
    )

    completed_items: Final = {
        event.item.call_id: event.item
        for event in iterator._pending_tool_events
        if event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
    }

    assert completed_items["call_function"].type == "function_call"
    assert completed_items["call_function"].arguments == '{"path":"/tmp"}'
    assert completed_items["call_custom"].type == "custom_tool_call"
    assert completed_items["call_custom"].name == "exec"
    assert completed_items["call_custom"].namespace == "shell"
    assert completed_items["call_custom"].input == "pwd"


def test_streaming_custom_wire_name_rejects_invalid_content_envelope():
    request: Final = {
        "tools": [
            {
                "type": "custom",
                "name": "exec",
                "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
            }
        ]
    }
    iterator: Final = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request=request,
    )
    tool_call: Final = {"id": "call_exec", "function": {"name": "exec", "arguments": "invalid"}}
    iterator._queue_tool_call_delta_events([tool_call])

    with pytest.raises(ValueError, match="valid JSON"):
        iterator._queue_final_tool_call_done_events(
            ModelResponse(
                id="chatcmpl-custom",
                created=1,
                model="test-model",
                object="chat.completion",
                choices=[
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                    }
                ],
            )
        )


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("arguments", ('{"content":""}', "invalid"))
@pytest.mark.asyncio
async def test_hosted_exec_grammar_rejects_invalid_input_before_custom_done(sync_mode: bool, arguments: str):
    request: Final = {
        "tools": [
            {
                "type": "custom",
                "name": "exec",
                "format": {"type": "grammar", "syntax": "lark", "definition": _HOSTED_EXEC_GRAMMAR},
            }
        ]
    }
    wire_name: Final = next(iter(native_responses_custom_tool_name_map(request)))
    iterator: Final = _build_iterator(
        [_custom_tool_call_chunk(wire_name, arguments, "call_exec", finish_reason="tool_calls")],
        custom_llm_provider="hosted_vllm",
        responses_api_request=request,
    )

    events, error = await _collect_events_until_bad_gateway(iterator, sync_mode)
    payloads: Final = tuple(event.model_dump(mode="json", exclude_none=True) for event in events)

    assert error.status_code == 502
    assert json.dumps(payloads, ensure_ascii=False)
    assert all(payload["type"] != ResponsesAPIStreamEvents.CUSTOM_TOOL_CALL_INPUT_DONE for payload in payloads)
    assert all(
        payload["type"] != ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
        or payload["item"]["type"] != "custom_tool_call"
        for payload in payloads
    )
    assert all(payload["type"] != ResponsesAPIStreamEvents.RESPONSE_COMPLETED for payload in payloads)


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_hosted_exec_grammar_preserves_fragmented_unicode_whitespace(sync_mode: bool):
    request: Final = {
        "tools": [
            {
                "type": "namespace",
                "name": "shell",
                "tools": [
                    {
                        "type": "custom",
                        "name": "exec",
                        "format": {"type": "grammar", "syntax": "lark", "definition": _HOSTED_EXEC_GRAMMAR},
                    }
                ],
            }
        ]
    }
    wire_name: Final = next(iter(native_responses_custom_tool_name_map(request)))
    custom_input: Final = "\tλ\n "
    arguments: Final = json.dumps({"content": custom_input, "extra": "ignored"}, ensure_ascii=False)
    split: Final = len(arguments) // 2
    iterator: Final = _build_iterator(
        [
            _custom_tool_call_chunk(wire_name, arguments[:split], "call_exec"),
            _custom_tool_call_chunk(wire_name, arguments[split:], None, finish_reason="tool_calls"),
        ],
        custom_llm_provider="hosted_vllm",
        responses_api_request=request,
    )

    events: Final = await _collect_events(iterator, sync_mode)
    payloads: Final = tuple(event.model_dump(mode="json", exclude_none=True) for event in events)
    input_done: Final = next(
        payload for payload in payloads if payload["type"] == ResponsesAPIStreamEvents.CUSTOM_TOOL_CALL_INPUT_DONE
    )
    item_done: Final = next(
        payload
        for payload in payloads
        if payload["type"] == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
        and payload["item"]["type"] == "custom_tool_call"
    )

    assert json.dumps(payloads, ensure_ascii=False)
    assert input_done["item_id"] == item_done["item"]["call_id"] == "call_exec"
    assert input_done["input"] == item_done["item"]["input"] == custom_input
    assert item_done["item"]["name"] == "exec"
    assert item_done["item"]["namespace"] == "shell"
    assert any(payload["type"] == ResponsesAPIStreamEvents.RESPONSE_COMPLETED for payload in payloads)


@pytest.mark.parametrize(
    "tool_format",
    (
        None,
        {"type": "text"},
        {"type": "grammar", "syntax": "lark", "definition": 'start: ""'},
    ),
)
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_hosted_nonexec_custom_tool_preserves_empty_custom_input(sync_mode: bool, tool_format):
    tool = {"type": "custom", "name": "write"}
    if tool_format is not None:
        tool["format"] = tool_format
    request: Final = {"tools": [tool]}
    wire_name: Final = next(iter(native_responses_custom_tool_name_map(request)))
    iterator: Final = _build_iterator(
        [
            _custom_tool_call_chunk(
                wire_name,
                '{"content":"","extra":"preserved"}',
                "call_write",
                finish_reason="tool_calls",
            )
        ],
        custom_llm_provider="hosted_vllm",
        responses_api_request=request,
    )

    events: Final = await _collect_events(iterator, sync_mode)
    payloads: Final = tuple(event.model_dump(mode="json", exclude_none=True) for event in events)
    input_done: Final = next(
        payload for payload in payloads if payload["type"] == ResponsesAPIStreamEvents.CUSTOM_TOOL_CALL_INPUT_DONE
    )

    assert json.dumps(payloads)
    assert input_done["input"] == ""
    assert any(payload["type"] == ResponsesAPIStreamEvents.RESPONSE_COMPLETED for payload in payloads)


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_tool_then_reasoning_then_text_gives_message_its_own_output_index(sync_mode: bool):
    iterator: Final = _build_iterator(
        [
            _tool_call_chunk(),
            _reasoning_chunk("thinking"),
            _chunk("Hello"),
            _chunk("!", finish_reason="stop"),
        ]
    )

    events: Final = await _collect_events(iterator, sync_mode)
    output_item_added_events: Final = [
        event for event in events if getattr(event, "type", None) == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    ]
    message_item_adds: Final = [event for event in output_item_added_events if _is_message_item(event)]
    function_call_adds: Final = [
        event for event in output_item_added_events if getattr(event.item, "type", None) == "function_call"
    ]

    assert len(message_item_adds) == 1
    assert all(message_item_adds[0].output_index != event.output_index for event in function_call_adds)

    output_indexes_by_item_id: Final = {event.item.id: event.output_index for event in output_item_added_events}
    assert len(output_indexes_by_item_id) == len(set(output_indexes_by_item_id.values()))


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_plain_text_stream_announces_exactly_one_message_item(sync_mode: bool):
    iterator: Final = _build_iterator([_chunk("Hel"), _chunk("lo", finish_reason="stop")])

    events: Final = await _collect_events(iterator, sync_mode)

    message_item_adds = [
        event
        for event in events
        if getattr(event, "type", None) == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED and _is_message_item(event)
    ]
    assert len(message_item_adds) == 1
    for event in events:
        if getattr(event, "type", None) in (
            ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
            ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
        ):
            assert event.item_id == message_item_adds[0].item.id
def test_custom_tool_stream_uses_custom_input_events():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={"tools": [{"type": "custom", "name": "exec"}]},
    )
    custom_input = "patch payload"
    arguments = json.dumps({"content": custom_input})

    iterator._queue_tool_call_delta_events(
        [
            {
                "id": "call_exec",
                "type": "function",
                "function": {"name": "exec", "arguments": arguments},
            }
        ]
    )
    iterator._queue_final_tool_call_done_events(
        ModelResponse(
            id="resp-1",
            created=123,
            model="test-model",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_exec",
                                "type": "function",
                                "function": {"name": "exec", "arguments": arguments},
                            }
                        ],
                    },
                }
            ],
        )
    )

    event_types = [event.type for event in iterator._pending_tool_events]
    assert event_types[0] == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    assert ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA not in event_types
    assert ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE not in event_types
    assert event_types[-2:] == [
        ResponsesAPIStreamEvents.CUSTOM_TOOL_CALL_INPUT_DONE,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
    ]
    assert iterator._pending_tool_events[-2].input == custom_input
    assert iterator._pending_tool_events[-1].item.input == custom_input


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_stream_metadata_matches_request_in_all_response_events(sync_mode: bool):
    request: Final = {
        "reasoning": {"effort": "medium"}, "metadata": {"fixture": "correlation"},
        "max_output_tokens": 128, "parallel_tool_calls": False, "store": False,
        "previous_response_id": "resp_fixture", "instructions": "Fixture instructions",
    }
    iterator: Final = LiteLLMCompletionStreamingIterator(
        model="fixture", litellm_custom_stream_wrapper=_FakeStreamWrapper([_chunk("answer", finish_reason="stop")]),
        request_input="Fixture", responses_api_request=request, custom_llm_provider="hosted_vllm",
    )
    events: Final = await _collect_events(iterator, sync_mode)
    responses: Final = [event.model_dump(mode="json")["response"] for event in events
                       if event.type in RESPONSE_ID_EVENT_TYPES]
    assert len(responses) == 3
    for response in responses:
        for key, value in request.items():
            assert response[key] == value
