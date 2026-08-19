"""
Tests for hosted_vllm responses API support.

Regression test for: https://github.com/BerriAI/litellm/issues
Bug: client.responses.create() raised TypeError: 'NoneType' object is not a mapping
when extra_body=None was passed through the responses→completion pipeline for
hosted_vllm (and any OpenAI-compatible provider using add_provider_specific_params_to_optional_params).
"""

import json
from functools import partial
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.hosted_vllm.responses.transformation import (
    HostedVLLMResponsesAPIConfig,
)
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

NATIVE_REASONING_CONFIG = {
    "levels": {"high": ["medium", "high"], "max": ["xhigh", "max"]},
    "disabled": "reject",
}
NATIVE_DEFAULT_REASONING_CONFIG = {}


def _make_mock_responses_api_response(content: str = "Hello! I'm doing well.") -> dict:
    return {
        "id": "resp-test123",
        "object": "response",
        "created_at": 1234567890,
        "model": "Qwen/Qwen3-8B",
        "output": [
            {
                "type": "message",
                "id": "msg-test123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        ],
        "status": "completed",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
    }


def _make_mock_http_client(response_body: dict) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = response_body
    mock_response.text = json.dumps(response_body)
    mock_client.post.return_value = mock_response
    return mock_client


def test_hosted_vllm_responses_create_with_string_input():
    """
    Test that hosted_vllm routes directly to the native /v1/responses endpoint
    when the Responses API config is registered, and correctly parses the response.
    """
    mock_client = _make_mock_http_client(_make_mock_responses_api_response("I'm doing well, thanks!"))

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
        return_value=mock_client,
    ):
        response = litellm.responses(
            model="hosted_vllm/Qwen/Qwen3-8B",
            input="Hello, how are you?",
            api_base="https://test-vllm.example.com/v1",
            api_key="test-key",
        )

    from litellm.types.llms.openai import ResponsesAPIResponse

    assert response is not None
    assert isinstance(response, ResponsesAPIResponse)
    assert len(response.output) > 0
    output_message = response.output[0]
    assert output_message.role == "assistant"  # type: ignore[union-attr]
    assert len(output_message.content) > 0  # type: ignore[union-attr]
    assert "well" in output_message.content[0].text  # type: ignore[union-attr]


def test_hosted_vllm_responses_create_with_explicit_none_extra_body():
    """
    Directly verify the fix in add_provider_specific_params_to_optional_params:
    extra_body=None must not crash when building optional_params.
    """
    from litellm.utils import get_optional_params

    # This should not raise TypeError: 'NoneType' object is not a mapping
    optional_params = get_optional_params(
        model="Qwen/Qwen3-8B",
        custom_llm_provider="hosted_vllm",
        extra_body=None,
    )

    # extra_body=None should be normalized to an empty dict (or absent)
    assert optional_params.get("extra_body") is not None or "extra_body" not in optional_params


def test_hosted_vllm_provider_config_registration():
    """Test that ProviderConfigManager returns HostedVLLMResponsesAPIConfig for hosted_vllm."""
    config = ProviderConfigManager.get_provider_responses_api_config(
        model="hosted_vllm/Qwen/Qwen3-8B",
        provider=LlmProviders.HOSTED_VLLM,
    )

    assert config is not None
    assert isinstance(config, HostedVLLMResponsesAPIConfig)
    assert config.custom_llm_provider == LlmProviders.HOSTED_VLLM


def test_hosted_vllm_responses_api_url():
    """Test get_complete_url() constructs the correct URL."""
    config = HostedVLLMResponsesAPIConfig()

    # api_base without /v1
    url = config.get_complete_url(
        api_base="http://localhost:8000",
        litellm_params={},
    )
    assert url == "http://localhost:8000/v1/responses"

    # api_base with /v1
    url_with_v1 = config.get_complete_url(
        api_base="http://localhost:8000/v1",
        litellm_params={},
    )
    assert url_with_v1 == "http://localhost:8000/v1/responses"

    # api_base with trailing slash
    url_with_slash = config.get_complete_url(
        api_base="http://localhost:8000/v1/",
        litellm_params={},
    )
    assert url_with_slash == "http://localhost:8000/v1/responses"


def test_hosted_vllm_responses_api_url_requires_api_base():
    """Test get_complete_url() raises ValueError when api_base is not set."""
    config = HostedVLLMResponsesAPIConfig()

    with pytest.raises(ValueError, match="api_base not set"):
        config.get_complete_url(
            api_base=None,
            litellm_params={},
        )


def test_hosted_vllm_validate_environment_default_api_key():
    """Test validate_environment() defaults to 'fake-api-key' when no key is provided."""
    config = HostedVLLMResponsesAPIConfig()

    headers = config.validate_environment(
        headers={},
        model="Qwen/Qwen3-8B",
        litellm_params=GenericLiteLLMParams(),
    )

    assert headers.get("Authorization") == "Bearer fake-api-key"


def test_hosted_vllm_validate_environment_custom_api_key():
    """Test validate_environment() uses the provided api_key."""
    config = HostedVLLMResponsesAPIConfig()

    headers = config.validate_environment(
        headers={},
        model="Qwen/Qwen3-8B",
        litellm_params=GenericLiteLLMParams(api_key="my-custom-key"),
    )

    assert headers.get("Authorization") == "Bearer my-custom-key"


@pytest.mark.parametrize(
    "optional_params, reasoning_config, expected",
    [
        (
            {"reasoning": {"effort": "xhigh", "summary": "auto"}},
            NATIVE_REASONING_CONFIG,
            {"reasoning": {"effort": "max", "summary": "auto"}},
        ),
        (
            {},
            NATIVE_DEFAULT_REASONING_CONFIG,
            {"reasoning": {"effort": "high"}},
        ),
        (
            {"reasoning": {"effort": "none"}},
            NATIVE_DEFAULT_REASONING_CONFIG,
            {"reasoning": {"effort": "none"}},
        ),
    ],
)
def test_hosted_vllm_responses_reasoning_config(optional_params, reasoning_config, expected):
    request = HostedVLLMResponsesAPIConfig().transform_responses_api_request(
        model="model",
        input="hello",
        response_api_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(model_info={"reasoning_effort": reasoning_config}),
        headers={},
    )

    assert {key: request[key] for key in expected} == expected
    assert "chat_template_kwargs" not in request


def test_hosted_vllm_responses_rejects_disabling_native_reasoning():
    with pytest.raises(litellm.UnsupportedParamsError, match="always has reasoning enabled"):
        HostedVLLMResponsesAPIConfig().transform_responses_api_request(
            model="model",
            input="hello",
            response_api_optional_request_params={"reasoning": {"effort": "none"}},
            litellm_params=GenericLiteLLMParams(model_info={"reasoning_effort": NATIVE_REASONING_CONFIG}),
            headers={},
        )


def _responses_sse(events: list[dict[str, object]]) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def _response_snapshot(response_id: str, status: str, output: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "model": "fixture",
        "status": status,
        "output": output,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "parallel_tool_calls": True,
        "temperature": None,
        "tool_choice": "auto",
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


def _completed_response(response_id: str, output: list[dict[str, object]]) -> dict[str, object]:
    return {"type": "response.completed", "response": _response_snapshot(response_id, "completed", output)}


async def _responses_stream_with_mock_transport(
    request: dict[str, object], response_events: list[dict[str, object]]
) -> tuple[list[object], list[dict[str, object]]]:
    requests: list[dict[str, object]] = []

    def respond(http_request: httpx.Request) -> httpx.Response:
        assert http_request.url.path == "/v1/responses"
        requests.append(json.loads(http_request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_responses_sse(response_events),
        )

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        stream: Final = await litellm.aresponses(**request, client=handler)
        return [event async for event in stream], requests
    finally:
        await handler.client.aclose()


@pytest.mark.asyncio
async def test_hosted_vllm_guardian_custom_tool_uses_native_responses_stream():
    tool_input: Final = "const output = await tools.exec_command({ cmd: 'true' });"
    function_call: Final = {
        "type": "function_call",
        "id": "fc_guardian_exec",
        "call_id": "call_guardian_exec",
        "name": "exec",
        "arguments": json.dumps({"content": tool_input}),
        "status": "completed",
    }
    response_events: Final = [
        {"type": "response.created", "response": _response_snapshot("resp_guardian", "in_progress", [])},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**function_call, "arguments": "", "status": "in_progress"},
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_guardian_exec",
            "output_index": 0,
            "delta": function_call["arguments"],
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_guardian_exec",
            "output_index": 0,
            "arguments": function_call["arguments"],
        },
        {"type": "response.output_item.done", "output_index": 0, "item": function_call},
        _completed_response("resp_guardian", [{**function_call, "id": "fc_final", "call_id": "call_final"}]),
    ]
    request: Final = {
        "model": "hosted_vllm/fixture",
        "input": "Return a fixture decision.",
        "api_base": "https://fixture.invalid/v1",
        "api_key": "fixture",
        "stream": True,
        "store": False,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": "low"},
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "guardian_fixture",
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": {"outcome": {"type": "string", "enum": ["allow", "deny"]}},
                    "required": ["outcome"],
                    "additionalProperties": False,
                },
            },
        },
        "tools": [
            {
                "type": "custom",
                "name": "exec",
                "format": {"type": "grammar", "syntax": "lark", "definition": "start: /[\\s\\S]+/"},
            },
            {
                "type": "function",
                "name": "wait",
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": {"cell_id": {"type": "string"}},
                    "required": ["cell_id"],
                    "additionalProperties": False,
                },
            },
        ],
    }

    events, requests = await _responses_stream_with_mock_transport(request, response_events)

    assert len(requests) == 1
    upstream: Final = requests[0]
    assert upstream["stream"] is True
    assert upstream["store"] is False
    assert upstream["reasoning"] == {"effort": "low"}
    assert upstream["text"] == request["text"]
    assert [tool["type"] for tool in upstream["tools"]] == ["function", "function"]
    assert upstream["tools"][0]["name"] == "exec"
    assert upstream["tools"][0]["parameters"]["required"] == ["content"]
    assert upstream["tools"][1]["name"] == "wait"
    event_types: Final = [
        getattr(getattr(event, "type", None), "value", getattr(event, "type", None)) for event in events
    ]
    assert "response.custom_tool_call_input.delta" in event_types
    assert "response.custom_tool_call_input.done" in event_types
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
    assert completed.output[0].call_id == "call_guardian_exec"
    assert completed.output[0].input == tool_input
    assert completed.usage.total_tokens == 5


@pytest.mark.asyncio
async def test_hosted_vllm_custom_history_preserves_function_and_namespace_tools():
    namespace_function_call: Final = {
        "type": "function_call",
        "id": "fc_agents",
        "call_id": "call_agents",
        "name": "collaboration__list_agents",
        "arguments": json.dumps({"path_prefix": "root"}),
        "status": "completed",
    }
    request: Final = {
        "model": "hosted_vllm/fixture",
        "input": [
            {
                "type": "custom_tool_call",
                "id": "ctc_fixture",
                "call_id": "call_fixture",
                "name": "exec",
                "input": "status",
            },
            {"type": "custom_tool_call_output", "call_id": "call_fixture", "output": "healthy"},
            {"role": "user", "content": "Continue the fixture."},
        ],
        "api_base": "https://fixture.invalid/v1",
        "api_key": "fixture",
        "stream": True,
        "tools": [
            {
                "type": "custom",
                "name": "exec",
                "format": {"type": "grammar", "syntax": "lark", "definition": "start: /[\\s\\S]+/"},
            },
            {
                "type": "function",
                "name": "wait",
                "parameters": {"type": "object", "properties": {"cell_id": {"type": "string"}}},
            },
            {
                "type": "namespace",
                "name": "collaboration",
                "tools": [
                    {
                        "type": "function",
                        "name": "list_agents",
                        "parameters": {"type": "object", "properties": {"path_prefix": {"type": "string"}}},
                    }
                ],
            },
            {"type": "web_search"},
        ],
    }
    response_events: Final = [
        {"type": "response.created", "response": _response_snapshot("resp_main", "in_progress", [])},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": namespace_function_call,
        },
        _completed_response("resp_main", [namespace_function_call]),
    ]

    events, requests = await _responses_stream_with_mock_transport(request, response_events)

    assert len(requests) == 1
    upstream: Final = requests[0]
    assert upstream["stream"] is True
    assert upstream["input"][0] == {
        "type": "function_call",
        "id": "ctc_fixture",
        "call_id": "call_fixture",
        "name": "exec",
        "arguments": json.dumps({"content": "status"}),
    }
    assert upstream["input"][1] == {"type": "function_call_output", "call_id": "call_fixture", "output": "healthy"}
    assert [tool["type"] for tool in upstream["tools"]] == ["function", "function", "function", "web_search"]
    assert upstream["tools"][1]["name"] == "wait"
    assert upstream["tools"][2]["name"] == "collaboration__list_agents"
    assert upstream["tools"][3] == {"type": "web_search"}
    completed: Final = next(
        event.response for event in events if getattr(event.type, "value", event.type) == "response.completed"
    )
    assert completed.output[0].type == "function_call"
    assert completed.output[0].name == "list_agents"
    assert completed.output[0].namespace == "collaboration"
    assert completed.usage.total_tokens == 5
    assert getattr(events[-1].type, "value", events[-1].type) == "response.completed"


@pytest.fixture
def developer_context():
    return [
        {"role": "system", "content": "Keep the existing system instruction."},
        {
            "type": "message",
            "role": "developer",
            "content": [
                {"type": "input_text", "text": "Retain skills instructions.\n完整内容"},
                {"type": "input_text", "text": "Retain permissions instructions."},
            ],
        },
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Agent context."}]},
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Coordination context."}]},
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "First user part."},
                {"type": "input_text", "text": "Second user part."},
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Give the task a short title and description."}],
        },
    ]


@pytest.fixture
def title_text_format():
    return {
        "format": {
            "type": "json_schema",
            "name": "codex_output_schema",
            "strict": True,
            "schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 36},
                    "description": {"type": "string", "minLength": 1},
                },
                "required": ["title", "description"],
            },
        },
    }


@pytest.mark.parametrize(
    "model_info", [None, {}, {"supports_developer_messages": True}, {"supports_developer_messages": False}]
)
@pytest.mark.parametrize("config_type", [HostedVLLMResponsesAPIConfig, OpenAIResponsesAPIConfig])
def test_developer_messages_only_map_for_opted_in_hosted_deployments(
    model_info, config_type, developer_context, title_text_format
):
    original = json.loads(json.dumps(developer_context))
    request = config_type().transform_responses_api_request(
        model="fixture",
        input=developer_context,
        response_api_optional_request_params={
            "instructions": "Separate system instructions.",
            "text": title_text_format,
        },
        litellm_params=GenericLiteLLMParams(model_info=model_info),
        headers={},
    )
    mapped = config_type is HostedVLLMResponsesAPIConfig and model_info == {"supports_developer_messages": False}
    expected = [{**item, "role": "system"} if mapped and item["role"] == "developer" else item for item in original]
    assert request["input"] == expected
    assert request["instructions"] == "Separate system instructions."
    assert request["text"] == title_text_format
    assert developer_context == original
    assert "supports_developer_messages" not in json.dumps(request)


@pytest.mark.parametrize(
    "input",
    [
        "Plain conversation input.",
        [
            {"role": "user", "content": "Question."},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Answer.", "annotations": []}],
                "id": "msg_previous",
                "status": "completed",
            },
            {"type": "message", "role": "developer", "content": "Mid-conversation guidance."},
            {"role": "user", "content": "Follow-up."},
        ],
    ],
)
def test_developer_mapping_preserves_ordinary_input_and_message_order(input):
    original = json.loads(json.dumps(input))
    request = HostedVLLMResponsesAPIConfig().transform_responses_api_request(
        model="fixture",
        input=input,
        response_api_optional_request_params={},
        litellm_params=GenericLiteLLMParams(model_info={"supports_developer_messages": False}),
        headers={},
    )
    expected = (
        original
        if isinstance(original, str)
        else [{**item, "role": "system"} if item["role"] == "developer" else item for item in original]
    )
    assert request["input"] == expected
    assert input == original


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_router_native_responses_maps_developer_with_tools(stream, developer_context, title_text_format):
    title = {"title": "Review changes", "description": "Review the proposed changes."}
    output = [
        {
            "type": "message",
            "id": "msg_title",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": json.dumps(title), "annotations": []}],
        }
    ]
    response = _response_snapshot("resp_title", "completed", output)
    received = []

    def respond(request):
        assert request.url.path == "/v1/responses"
        received.append(json.loads(request.content))
        if stream:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=_responses_sse(
                    [
                        {"type": "response.created", "response": _response_snapshot("resp_title", "in_progress", [])},
                        {"type": "response.completed", "response": response},
                    ]
                ),
            )
        return httpx.Response(200, json=response)

    handler = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    router = litellm.Router(
        model_list=[
            {
                "model_name": "fixture",
                "litellm_params": {
                    "model": "hosted_vllm/fixture",
                    "api_base": "https://fixture.invalid/v1",
                    "api_key": "fixture",
                },
                "model_info": {"supports_developer_messages": False},
            }
        ]
    )
    tools = [
        {"type": "custom", "name": "exec", "format": {"type": "text"}},
        {"type": "function", "name": "wait", "parameters": {"type": "object", "properties": {}}},
        {
            "type": "namespace",
            "name": "helpers",
            "tools": [{"type": "function", "name": "inspect", "parameters": {"type": "object", "properties": {}}}],
        },
    ]
    history = [
        {"type": "custom_tool_call", "call_id": "call_previous", "name": "exec", "input": "Synthetic input."},
        {"type": "custom_tool_call_output", "call_id": "call_previous", "output": "Synthetic result."},
    ]
    try:
        result = await router._aresponses_with_streaming_fallbacks(
            original_function=partial(litellm.aresponses, client=handler),
            model="fixture",
            input=[*history, *developer_context],
            instructions="Separate system instructions.",
            tools=tools,
            text=title_text_format,
            stream=stream,
            store=False,
            client=handler,
        )
        if stream:
            events = [event async for event in result]
            assert events[-1].type == "response.completed"
            completed = events[-1].response
        else:
            completed = result
    finally:
        await handler.client.aclose()
    assert completed.status == "completed"
    assert json.loads(completed.output[0].content[0].text) == title
    assert completed.usage.total_tokens == 5
    assert len(received) == 1
    upstream = received[0]
    assert upstream["stream"] is stream
    assert upstream["text"] == title_text_format
    assert upstream["instructions"] == "Separate system instructions."
    assert upstream["input"][2:] == [
        {**item, "role": "system"} if item["role"] == "developer" else item for item in developer_context
    ]
    assert upstream["input"][:2] == [
        {
            "type": "function_call",
            "call_id": "call_previous",
            "name": "exec",
            "arguments": json.dumps({"content": "Synthetic input."}),
        },
        {"type": "function_call_output", "call_id": "call_previous", "output": "Synthetic result."},
    ]
    assert [(tool["type"], tool["name"]) for tool in upstream["tools"]] == [
        ("function", "exec"),
        ("function", "wait"),
        ("function", "helpers__inspect"),
    ]
    assert "supports_developer_messages" not in json.dumps(upstream)
