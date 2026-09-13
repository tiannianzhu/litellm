import json
from contextlib import nullcontext
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.hosted_vllm.chat.transformation import (
    HostedVLLMChatConfig,
    HostedVLLMChatStreamingHandler,
)

NATIVE_REASONING_CONFIG = {
    "levels": {"high": ["medium", "high"], "max": ["xhigh", "max"]},
    "disabled": "reject",
}
NATIVE_DEFAULT_REASONING_CONFIG = {}


def test_hosted_vllm_chat_transformation_file_url():
    config = HostedVLLMChatConfig()
    video_url = "https://example.com/video.mp4"
    video_data = f"data:video/mp4;base64,{video_url}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "file_data": video_data,
                    },
                }
            ],
        }
    ]
    transformed_response = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert transformed_response["messages"] == [
        {
            "role": "user",
            "content": [{"type": "video_url", "video_url": {"url": video_data}}],
        }
    ]


def test_hosted_vllm_supports_reasoning_effort():
    config = HostedVLLMChatConfig()
    supported_params = config.get_supported_openai_params(model="hosted_vllm/gpt-oss-120b")
    assert "reasoning_effort" in supported_params
    optional_params = config.map_openai_params(
        non_default_params={"reasoning_effort": "high"},
        optional_params={},
        model="hosted_vllm/gpt-oss-120b",
        drop_params=False,
    )
    assert optional_params["reasoning_effort"] == "high"


def test_hosted_vllm_streaming_usage_only_chunk_is_unchanged():
    handler = HostedVLLMChatStreamingHandler(streaming_response=None, sync_stream=True)
    usage_chunk = {
        "id": "chatcmpl-usage",
        "object": "chat.completion.chunk",
        "created": 1771411455,
        "model": "test-model",
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    parsed_chunk = handler.chunk_parser(usage_chunk)

    assert parsed_chunk.choices == []
    assert parsed_chunk.usage.prompt_tokens == 10


@pytest.mark.asyncio
async def test_hosted_vllm_async_streaming_tool_call_finish_reason_is_consistent():
    chunks = (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-tool",
                "object": "chat.completion.chunk",
                "created": 1771411455,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Boston"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )
        + "\n\n",
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-tool",
                "object": "chat.completion.chunk",
                "created": 1771411455,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        + "\n\n",
        "data: [DONE]\n\n",
    )

    async def stream():
        for chunk in chunks:
            yield chunk

    handler = HostedVLLMChatStreamingHandler(
        streaming_response=stream(),
        sync_stream=False,
    )

    parsed_chunks = [chunk async for chunk in handler]

    assert parsed_chunks[0].choices[0].delta.tool_calls is not None
    assert parsed_chunks[0].choices[0].finish_reason is None
    assert parsed_chunks[1].choices[0].finish_reason == "tool_calls"


def test_hosted_vllm_sync_streaming_tool_call_finish_reason_is_consistent():
    chunks = (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-tool",
                "object": "chat.completion.chunk",
                "created": 1771411455,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Boston"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )
        + "\n\n",
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-tool",
                "object": "chat.completion.chunk",
                "created": 1771411455,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        + "\n\n",
        "data: [DONE]\n\n",
    )

    handler = HostedVLLMChatStreamingHandler(
        streaming_response=iter(chunks),
        sync_stream=True,
    )

    parsed_chunks = list(handler)

    assert parsed_chunks[0].choices[0].delta.tool_calls is not None
    assert parsed_chunks[0].choices[0].finish_reason is None
    assert parsed_chunks[1].choices[0].finish_reason == "tool_calls"
    assert parsed_chunks[2]["is_finished"] is True


def test_hosted_vllm_sync_streaming_text_finish_reason_is_unchanged():
    chunks = (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-text",
                "object": "chat.completion.chunk",
                "created": 1771411455,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "No tools here"},
                        "finish_reason": None,
                    }
                ],
            }
        )
        + "\n\n",
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-text",
                "object": "chat.completion.chunk",
                "created": 1771411455,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        + "\n\n",
        "data: [DONE]\n\n",
    )

    handler = HostedVLLMChatStreamingHandler(
        streaming_response=iter(chunks),
        sync_stream=True,
    )

    parsed_chunks = list(handler)

    assert parsed_chunks[0].choices[0].finish_reason is None
    assert parsed_chunks[1].choices[0].finish_reason == "stop"


def test_hosted_vllm_streaming_tool_call_finish_reason_is_corrected():
    handler = HostedVLLMChatStreamingHandler(streaming_response=None, sync_stream=True)
    tool_chunk = {
        "id": "chatcmpl-tool",
        "object": "chat.completion.chunk",
        "created": 1771411455,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-tool",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Boston"}',
                            },
                        }
                    ],
                },
                "finish_reason": None,
            }
        ],
    }
    stop_chunk = {
        "id": "chatcmpl-tool",
        "object": "chat.completion.chunk",
        "created": 1771411455,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }

    parsed_tool_chunk = handler.chunk_parser(tool_chunk)
    parsed_stop_chunk = handler.chunk_parser(stop_chunk)

    assert parsed_tool_chunk.choices[0].finish_reason is None
    assert parsed_stop_chunk.choices[0].finish_reason == "tool_calls"


def test_hosted_vllm_streaming_tool_call_and_stop_in_same_chunk():
    handler = HostedVLLMChatStreamingHandler(streaming_response=None, sync_stream=True)
    chunk = {
        "id": "chatcmpl-tool",
        "object": "chat.completion.chunk",
        "created": 1771411455,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]},
                "finish_reason": "stop",
            }
        ],
    }

    parsed_chunk = handler.chunk_parser(chunk)

    assert parsed_chunk.choices[0].finish_reason == "tool_calls"


def test_hosted_vllm_streaming_preserves_non_tool_finish_reasons():
    handler = HostedVLLMChatStreamingHandler(streaming_response=None, sync_stream=True)

    for finish_reason in ("length", "content_filter"):
        parsed_chunk = handler.chunk_parser(
            {
                "id": "chatcmpl-tool",
                "object": "chat.completion.chunk",
                "created": 1771411455,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
        )
        assert parsed_chunk.choices[0].finish_reason == finish_reason


def test_hosted_vllm_streaming_tool_call_state_is_per_choice():
    handler = HostedVLLMChatStreamingHandler(streaming_response=None, sync_stream=True)
    multi_choice_chunk = {
        "id": "chatcmpl-multi",
        "object": "chat.completion.chunk",
        "created": 1771411455,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "No tool here"},
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "delta": {"tool_calls": [{"index": 0, "function": {"name": "get_weather"}}]},
                "finish_reason": "stop",
            },
        ],
    }
    finish_chunk = {
        "id": "chatcmpl-multi",
        "object": "chat.completion.chunk",
        "created": 1771411455,
        "model": "test-model",
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"},
            {"index": 1, "delta": {}, "finish_reason": "stop"},
        ],
    }

    handler.chunk_parser(multi_choice_chunk)
    parsed_finish_chunk = handler.chunk_parser(finish_chunk)

    assert parsed_finish_chunk.choices[0].finish_reason == "stop"
    assert parsed_finish_chunk.choices[1].finish_reason == "tool_calls"


def test_hosted_vllm_streaming_empty_tool_list_keeps_stop():
    handler = HostedVLLMChatStreamingHandler(streaming_response=None, sync_stream=True)
    parsed_chunk = handler.chunk_parser(
        {
            "id": "chatcmpl-empty-tools",
            "object": "chat.completion.chunk",
            "created": 1771411455,
            "model": "test-model",
            "choices": [
                {"index": 0, "delta": {"tool_calls": []}, "finish_reason": "stop"},
            ],
        }
    )

    assert parsed_chunk.choices[0].finish_reason == "stop"


def _hosted_vllm_sse_chunk(choices, created, usage=None, system_fingerprint=None):
    payload = {
        "id": "chatcmpl-fixture",
        "object": "chat.completion.chunk",
        "created": created,
        "model": "fixture",
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    if system_fingerprint is not None:
        payload["system_fingerprint"] = system_fingerprint
    return f"data: {json.dumps(payload)}\n\n"


def _hosted_vllm_multi_choice_sse(finish_reason, tool_and_finish_share_chunk):
    tool_delta = {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_fixture",
                "type": "function",
                "function": {"name": "echo", "arguments": '{"strict":true}'},
            }
        ]
    }
    text_delta = {"content": "hello", "tool_calls": []}
    tool_choice = {
        "index": 1,
        "delta": tool_delta,
        "finish_reason": finish_reason if tool_and_finish_share_chunk else None,
    }
    text_choice = {"index": 0, "delta": text_delta, "finish_reason": None}
    text_finish = {"index": 0, "delta": {}, "finish_reason": "stop"}
    tool_finish = {"index": 1, "delta": {}, "finish_reason": finish_reason}
    first_chunk = _hosted_vllm_sse_chunk([tool_choice, text_choice], created=1, system_fingerprint="fp-fixture")
    usage_chunk = _hosted_vllm_sse_chunk(
        [], created=3, usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    )
    return (
        first_chunk
        + _hosted_vllm_sse_chunk([text_finish], created=2)
        + ("" if tool_and_finish_share_chunk else _hosted_vllm_sse_chunk([tool_finish], created=4))
        + usage_chunk
        + "data: [DONE]\n\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "contents, should_raise",
    [(("repeat",) * 5, True), (("ab",) * 5, False), (("one", "two", "three", "four", "five"), False)],
    ids=["repetition", "short-tokens", "distinct-text"],
)
async def test_hosted_vllm_streaming_repetition_guard(
    monkeypatch: pytest.MonkeyPatch, async_mode: bool, contents: tuple[str, ...], should_raise: bool
) -> None:
    monkeypatch.setattr(litellm, "REPEATED_STREAMING_CHUNK_LIMIT", 3)
    stream_body: Final = "".join(
        _hosted_vllm_sse_chunk([{"index": 0, "delta": {"content": content}, "finish_reason": None}], created=1)
        for content in contents
    ) + _hosted_vllm_sse_chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}], created=1) + "data: [DONE]\n\n"

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=stream_body)

    request: Final = {
        "model": "hosted_vllm/fixture",
        "messages": [{"role": "user", "content": "echo"}],
        "api_base": "https://fixture.invalid/v1",
        "api_key": "fixture",
        "stream": True,
    }
    expectation: Final = (
        pytest.raises(litellm.exceptions.MidStreamFallbackError, match="repeating the same chunk")
        if should_raise
        else nullcontext()
    )
    if async_mode:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as async_transport:
            async_handler: Final = AsyncHTTPHandler()
            await async_handler.client.aclose()
            async_handler.client = async_transport
            async_stream: Final = await litellm.acompletion(**request, client=async_handler)
            with expectation:
                async_chunks: Final = [chunk async for chunk in async_stream]
                assert "".join(chunk.choices[0].delta.content or "" for chunk in async_chunks) == "".join(contents)
    else:
        with httpx.Client(transport=httpx.MockTransport(respond)) as sync_transport:
            sync_stream: Final = litellm.completion(**request, client=HTTPHandler(client=sync_transport))
            with expectation:
                sync_chunks: Final = list(sync_stream)
                assert "".join(chunk.choices[0].delta.content or "" for chunk in sync_chunks) == "".join(contents)


def _stream_choice_sequence(chunks):
    return [
        [
            (choice.index, choice.delta.content, bool(choice.delta.tool_calls), choice.finish_reason)
            for choice in chunk.choices
        ]
        for chunk in chunks
    ]


def _assert_hosted_vllm_multi_choice_stream(chunks, finish_reason, tool_and_finish_share_chunk):
    tool_finish_reason = "tool_calls" if finish_reason == "stop" else finish_reason
    assert _stream_choice_sequence(chunks) == [
        [(1, None, True, tool_finish_reason if tool_and_finish_share_chunk else None), (0, "hello", False, None)],
        [(0, None, False, "stop")],
    ] + ([] if tool_and_finish_share_chunk else [[(1, None, False, tool_finish_reason)]]) + [[]]
    assert [usage.total_tokens for chunk in chunks if (usage := getattr(chunk, "usage", None)) is not None] == [3]
    assert {chunk.created for chunk in chunks} == {chunks[0].created}
    assert [chunk.system_fingerprint for chunk in chunks] == ["fp-fixture"] * len(chunks)


@pytest.mark.parametrize(
    "finish_reason, tool_and_finish_share_chunk",
    [
        ("stop", True),
        ("stop", False),
        ("length", False),
        ("content_filter", False),
    ],
)
def test_hosted_vllm_sync_multi_choice_stream_keeps_every_terminal(finish_reason, tool_and_finish_share_chunk):
    stream_body = _hosted_vllm_multi_choice_sse(finish_reason, tool_and_finish_share_chunk)

    def respond(_request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=stream_body)

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        stream = litellm.completion(
            model="hosted_vllm/fixture",
            messages=[{"role": "user", "content": "echo"}],
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            stream=True,
            stream_options={"include_usage": True},
            n=2,
            client=HTTPHandler(client=transport),
        )
        chunks = list(stream)

    _assert_hosted_vllm_multi_choice_stream(chunks, finish_reason, tool_and_finish_share_chunk)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish_reason, tool_and_finish_share_chunk",
    [
        ("stop", True),
        ("stop", False),
        ("length", False),
        ("content_filter", False),
    ],
)
async def test_hosted_vllm_async_multi_choice_stream_keeps_every_terminal(finish_reason, tool_and_finish_share_chunk):
    stream_body = _hosted_vllm_multi_choice_sse(finish_reason, tool_and_finish_share_chunk)

    async def respond(_request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=stream_body)

    client = AsyncHTTPHandler()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        stream = await litellm.acompletion(
            model="hosted_vllm/fixture",
            messages=[{"role": "user", "content": "echo"}],
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            stream=True,
            stream_options={"include_usage": True},
            n=2,
            client=client,
        )
        chunks = [chunk async for chunk in stream]
    finally:
        await client.client.aclose()

    _assert_hosted_vllm_multi_choice_stream(chunks, finish_reason, tool_and_finish_share_chunk)


def test_hosted_vllm_supports_thinking():
    """
    Test that hosted_vllm supports the 'thinking' parameter.

    Anthropic-style thinking is converted to OpenAI-style reasoning_effort
    since vLLM is OpenAI-compatible.

    Related issue: https://github.com/BerriAI/litellm/issues/19761
    """
    config = HostedVLLMChatConfig()
    supported_params = config.get_supported_openai_params(model="hosted_vllm/GLM-4.6-FP8")
    assert "thinking" in supported_params

    # Test thinking below the low threshold -> "minimal"
    optional_params = config.map_openai_params(
        non_default_params={
            "thinking": {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET - 1,
            }
        },
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert "thinking" not in optional_params  # thinking should NOT be passed
    assert optional_params["reasoning_effort"] == "minimal"

    # Test thinking with high budget_tokens -> "high"
    optional_params = config.map_openai_params(
        non_default_params={
            "thinking": {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
            }
        },
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert optional_params["reasoning_effort"] == "high"

    # Test that existing reasoning_effort is not overwritten
    optional_params = config.map_openai_params(
        non_default_params={
            "thinking": {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
            },
            "reasoning_effort": "low",
        },
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert optional_params["reasoning_effort"] == "low"


def test_hosted_vllm_thinking_blocks_prepended_to_assistant_content():
    """
    Test that thinking_blocks on assistant messages are removed and content
    stays a string for vLLM compatibility.
    """
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Here is my answer.",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Let me reason about this...",
                    "signature": "abc123",
                }
            ],
            "reasoning_content": "Let me reason about this...",
        },
        {
            "role": "user",
            "content": "Follow up question",
        },
    ]
    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assistant_msg = transformed["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert isinstance(assistant_msg["content"], str)
    assert assistant_msg["content"] == "Here is my answer."
    assert "thinking_blocks" not in assistant_msg
    assert "reasoning_content" not in assistant_msg


def test_hosted_vllm_thinking_blocks_with_list_content():
    """
    Test thinking_blocks are removed and assistant content list is converted
    to a string.
    """
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Response text"}],
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Step 1 reasoning",
                    "signature": "sig1",
                },
                {
                    "type": "thinking",
                    "thinking": "Step 2 reasoning",
                    "signature": "sig2",
                },
            ],
        },
    ]
    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assistant_msg = transformed["messages"][0]
    assert isinstance(assistant_msg["content"], str)
    assert assistant_msg["content"] == "Response text"
    assert "thinking_blocks" not in assistant_msg


def test_hosted_vllm_assistant_structured_content_is_preserved():
    config = HostedVLLMChatConfig()
    image_block = {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image.png"},
    }
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Here is the image"}, image_block],
        },
    ]

    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assistant_msg = transformed["messages"][0]
    assert assistant_msg["content"] == [
        {"type": "text", "text": "Here is the image"},
        image_block,
    ]


def test_hosted_vllm_assistant_tool_use_content_becomes_tool_calls():
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Boston"},
                }
            ],
        },
    ]

    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assistant_msg = transformed["messages"][0]
    assert assistant_msg["content"] == ""
    assert assistant_msg["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Boston"}),
            },
        }
    ]


def test_hosted_vllm_assistant_tool_use_does_not_duplicate_existing_tool_calls():
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Boston"},
                }
            ],
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Boston"}),
                    },
                }
            ],
        },
    ]

    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assistant_msg = transformed["messages"][0]
    assert assistant_msg["content"] == ""
    assert assistant_msg["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Boston"}),
            },
        }
    ]


def test_hosted_vllm_custom_tools_are_converted_to_function_tools():
    config = HostedVLLMChatConfig()
    optional_params = config.map_openai_params(
        non_default_params={
            "tools": [
                {
                    "type": "custom",
                    "custom": {
                        "name": "apply_patch",
                        "description": "Apply text patch",
                        "format": {
                            "type": "grammar",
                            "grammar": {"syntax": "lark", "definition": "start: /.*/"},
                        },
                    },
                }
            ]
        },
        optional_params={},
        model="hosted_vllm/gpt-oss-120b",
        drop_params=False,
    )

    tools = optional_params["tools"]
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "apply_patch"
    assert tools[0]["function"]["description"] == "Apply text patch"
    assert tools[0]["function"]["parameters"]["type"] == "object"
    assert "input" in tools[0]["function"]["parameters"]["properties"]


def test_hosted_vllm_custom_tools_use_top_level_input_schema():
    config = HostedVLLMChatConfig()
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    optional_params = config.map_openai_params(
        non_default_params={
            "tools": [
                {
                    "type": "custom",
                    "name": "search",
                    "description": "Search docs",
                    "input_schema": input_schema,
                }
            ]
        },
        optional_params={},
        model="hosted_vllm/gpt-oss-120b",
        drop_params=False,
    )

    tools = optional_params["tools"]
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "search"
    assert tools[0]["function"]["description"] == "Search docs"
    assert tools[0]["function"]["parameters"] == input_schema


@pytest.mark.parametrize(
    "params, optional_params, expected",
    [
        (
            {"reasoning_effort": "xhigh", "reasoning_effort_config": NATIVE_REASONING_CONFIG},
            {},
            {"reasoning_effort": "max"},
        ),
        (
            {"reasoning_effort_config": NATIVE_DEFAULT_REASONING_CONFIG},
            {},
            {"reasoning_effort": "high"},
        ),
        (
            {"reasoning_effort": "medium", "reasoning_effort_config": NATIVE_DEFAULT_REASONING_CONFIG},
            {},
            {"reasoning_effort": "high"},
        ),
        (
            {"reasoning_effort": "xhigh", "reasoning_effort_config": NATIVE_DEFAULT_REASONING_CONFIG},
            {},
            {"reasoning_effort": "max"},
        ),
        (
            {"reasoning_effort": "none", "reasoning_effort_config": NATIVE_DEFAULT_REASONING_CONFIG},
            {},
            {"reasoning_effort": "none"},
        ),
        (
            {"thinking": {"type": "enabled", "budget_tokens": 1}, "reasoning_effort": "max"},
            {},
            {"reasoning_effort": "max"},
        ),
    ],
)
def test_hosted_vllm_reasoning_config(params, optional_params, expected):
    result = HostedVLLMChatConfig().map_openai_params(
        non_default_params=params,
        optional_params=optional_params,
        model="hosted_vllm/model",
        drop_params=False,
    )

    assert result == expected


def test_hosted_vllm_reasoning_config_rejects_disabling_native_reasoning():
    with pytest.raises(litellm.UnsupportedParamsError, match="always has reasoning enabled"):
        HostedVLLMChatConfig().map_openai_params(
            non_default_params={"reasoning_effort": "none", "reasoning_effort_config": NATIVE_REASONING_CONFIG},
            optional_params={},
            model="hosted_vllm/model",
            drop_params=False,
        )


@pytest.mark.parametrize("default", [None, "max"])
def test_hosted_vllm_rejects_model_specific_reasoning_defaults(default):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        HostedVLLMChatConfig().map_openai_params(
            non_default_params={"reasoning_effort_config": {"default": default}},
            optional_params={},
            model="hosted_vllm/model",
            drop_params=False,
        )
