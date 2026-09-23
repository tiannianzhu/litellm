import json

import httpx
import pytest

import litellm
from litellm.llms.hosted_vllm.messages.transformation import (
    HostedVLLMAnthropicMessagesConfig,
    _correct_vllm_messages_stream,
)
from litellm.types.router import GenericLiteLLMParams

NATIVE_CONFIG = {
    "levels": {"low": ["minimal", "low"], "high": ["medium", "high"], "max": ["xhigh", "max"]},
    "disabled": "reject",
}
SWITCHABLE_NATIVE_CONFIG = {}


def transform(optional_params, reasoning_config=SWITCHABLE_NATIVE_CONFIG, **litellm_params):
    return HostedVLLMAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 1024, **optional_params},
        litellm_params=GenericLiteLLMParams(model_info={"reasoning_effort": reasoning_config}, **litellm_params),
        headers={},
    )


def test_hosted_vllm_messages_rejects_disabling_native_reasoning():
    with pytest.raises(litellm.UnsupportedParamsError, match="always has reasoning enabled"):
        transform({"thinking": {"type": "disabled"}}, reasoning_config=NATIVE_CONFIG)


@pytest.mark.parametrize(
    "optional_params, expected",
    [
        (
            {"thinking": {"type": "disabled"}, "output_config": {"effort": "high"}},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ),
        (
            {"thinking": {"type": "adaptive"}},
            {
                "chat_template_kwargs": {"enable_thinking": True},
                "output_config": {"effort": "high"},
            },
        ),
        (
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}},
            {
                "chat_template_kwargs": {"enable_thinking": True},
                "output_config": {"effort": "max"},
            },
        ),
        (
            {"output_config": {"effort": "high"}},
            {"output_config": {"effort": "high"}},
        ),
        (
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            {"chat_template_kwargs": {"enable_thinking": True}, "output_config": {"effort": "low"}},
        ),
        (
            {"output_config": {"effort": "medium", "format": {"type": "json"}}},
            {"output_config": {"effort": "high", "format": {"type": "json"}}},
        ),
        (
            {},
            {"output_config": {"effort": "high"}},
        ),
    ],
)
def test_hosted_vllm_messages_native_switch_mapping(optional_params, expected):
    request = transform(optional_params, reasoning_config=SWITCHABLE_NATIVE_CONFIG)

    assert {key: request[key] for key in expected} == expected
    assert request.get("output_config", {}).get("effort") != "none"
    assert "thinking" not in request
    if optional_params.get("thinking", {}).get("type") == "disabled":
        assert "effort" not in request.get("output_config", {})


@pytest.mark.parametrize(
    "content, upstream_reason, expected_reason",
    [
        ([{"type": "tool_use", "id": "tool_1", "name": "echo", "input": {}}], "end_turn", "tool_use"),
        ([{"type": "text", "text": "Done"}], "end_turn", "end_turn"),
        ([{"type": "tool_use", "id": "tool_1", "name": "echo", "input": {}}], "max_tokens", "max_tokens"),
        ([{"type": "tool_use", "id": "tool_1", "name": "echo", "input": {}}], "tool_use", "tool_use"),
    ],
)
def test_hosted_vllm_messages_corrects_only_generated_tool_use(content, upstream_reason, expected_reason):
    upstream = httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "model",
            "content": content,
            "stop_reason": upstream_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )

    result = HostedVLLMAnthropicMessagesConfig().transform_anthropic_messages_response(
        model="model", raw_response=upstream, logging_obj=None
    )

    assert result["stop_reason"] == expected_reason
    assert result["content"] == content


def _sse_event(event):
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block_type, upstream_reason, expected_reason",
    [
        ("tool_use", "end_turn", "tool_use"),
        ("text", "end_turn", "end_turn"),
        ("tool_use", "max_tokens", "max_tokens"),
        ("tool_use", "tool_use", "tool_use"),
    ],
)
async def test_hosted_vllm_messages_stream_corrects_only_generated_tool_use(
    block_type, upstream_reason, expected_reason
):
    start = _sse_event({"type": "content_block_start", "index": 0, "content_block": {"type": block_type}})
    delta = _sse_event(
        {"type": "message_delta", "delta": {"stop_reason": upstream_reason}, "usage": {"output_tokens": 2}}
    )
    stop = _sse_event({"type": "message_stop"})
    upstream = start + delta + stop

    async def chunks():
        yield upstream[:17]
        yield upstream[17 : len(start) + 12]
        yield upstream[len(start) + 12 :]

    result = b"".join([chunk async for chunk in _correct_vllm_messages_stream(chunks())])
    events = [json.loads(frame.split(b"data:", 1)[1]) for frame in result.split(b"\n\n") if frame]

    assert events[0]["content_block"]["type"] == block_type
    assert events[1]["delta"]["stop_reason"] == expected_reason
    assert events[1]["usage"] == {"output_tokens": 2}
    assert events[2]["type"] == "message_stop"
    if expected_reason == upstream_reason:
        assert result == upstream
