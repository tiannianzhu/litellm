"""
Tests for forcing the /responses → /chat/completions bridge for `openai/` models
(via `use_chat_completions_api` or the `openai/chat_completions/<model>` model id).

Includes file_search emulation: the flag must be forwarded on inner aresponses
calls so routed requests do not hit a custom api_base /v1/responses endpoint.
"""

import json
from importlib import import_module
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import Choices, Message, ModelResponse, Usage


class TestUseResponsesApiBridgeFlag:
    """Test that bridge opt-in forces the chat completions path."""

    @pytest.mark.parametrize("model", ["openai/chat_completions/gpt-6-astra", "xai/test-classifier"])
    def test_encrypted_classifier_rejection_preserves_public_error(self, model: str) -> None:
        respond: Final = MagicMock(side_effect=AssertionError("Incompatible classifier sent an upstream request"))
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(
                litellm.APIConnectionError,
                match="Encrypted task classification requires a compatible native Responses deployment",
            ) as error:
                litellm.responses(
                    model=model,
                    input="Delegated task",
                    api_key="test-key",
                    api_base="https://classifier.test/v1",
                    client=HTTPHandler(client=client),
                    _require_encrypted_task_support=True,
                    num_retries=0,
                )
        assert error.value.status_code == 500
        respond.assert_not_called()

    @patch.object(
        import_module("litellm.responses.main").litellm_completion_transformation_handler, "response_api_handler"
    )
    @patch.object(
        import_module("litellm.responses.main").ProviderConfigManager, "get_provider_responses_api_config"
    )
    def test_bridge_used_when_use_chat_completions_api_true(
        self, mock_get_config, mock_bridge_handler
    ):
        """When use_chat_completions_api=True, the bridge handler should be called."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/my-custom-model",
            input="Hello",
            use_chat_completions_api=True,
            litellm_logging_obj=MagicMock(),
        )

        mock_bridge_handler.assert_called_once()

    @patch.object(
        import_module("litellm.responses.main").litellm_completion_transformation_handler, "response_api_handler"
    )
    @patch.object(
        import_module("litellm.responses.main").ProviderConfigManager, "get_provider_responses_api_config"
    )
    def test_provider_affinity_header_is_forwarded_through_bridge(self, mock_get_config, mock_bridge_handler):
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/my-custom-model",
            input="Hello",
            use_chat_completions_api=True,
            litellm_session_id="session-bridge",
            provider_affinity_header="X-Conversation-Id",
            extra_headers={"X-Customer-Header": "customer-value"},
            litellm_logging_obj=MagicMock(),
        )

        forwarded_headers = mock_bridge_handler.call_args.kwargs["extra_headers"]
        assert forwarded_headers["X-Conversation-Id"] == "session-bridge"
        assert forwarded_headers["X-Customer-Header"] == "customer-value"

    @patch.object(
        import_module("litellm.responses.main").litellm_completion_transformation_handler, "response_api_handler"
    )
    @patch.object(
        import_module("litellm.responses.main").ProviderConfigManager, "get_provider_responses_api_config"
    )
    def test_bridge_used_when_model_uses_chat_completions_prefix(
        self, mock_get_config, mock_bridge_handler
    ):
        """`openai/chat_completions/<name>` normalizes to `openai/<name>` and uses the bridge."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/chat_completions/my-custom-model",
            input="Hello",
            litellm_logging_obj=MagicMock(),
        )

        mock_bridge_handler.assert_called_once()
        # Model string is provider-normalized after resolution; prefix only forces the bridge.
        assert mock_bridge_handler.call_args.kwargs["model"].endswith("my-custom-model")

    @patch.object(import_module("litellm.responses.main").base_llm_http_handler, "response_api_handler")
    @patch.object(
        import_module("litellm.responses.main").ProviderConfigManager, "get_provider_responses_api_config"
    )
    def test_native_forwarding_when_flag_absent(
        self, mock_get_config, mock_native_handler
    ):
        """When use_chat_completions_api is not set, openai/ models should use
        native responses API forwarding (existing behavior)."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_native_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/gpt-4o",
            input="Hello",
            litellm_logging_obj=MagicMock(),
        )

        mock_native_handler.assert_called_once()

    @patch.object(
        import_module("litellm.responses.main").litellm_completion_transformation_handler, "response_api_handler"
    )
    @patch.object(
        import_module("litellm.responses.main").ProviderConfigManager, "get_provider_responses_api_config"
    )
    def test_flag_does_not_leak_into_kwargs(self, mock_get_config, mock_bridge_handler):
        """use_chat_completions_api should be popped and not passed to the bridge handler."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/my-custom-model",
            input="Hello",
            use_chat_completions_api=True,
            litellm_logging_obj=MagicMock(),
        )

        call_kwargs = mock_bridge_handler.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        assert "use_chat_completions_api" not in all_kwargs

    @patch.object(
        import_module("litellm.responses.main").litellm_completion_transformation_handler, "response_api_handler"
    )
    @patch.object(
        import_module("litellm.responses.main").ProviderConfigManager, "get_provider_responses_api_config"
    )
    def test_bridge_used_when_provider_config_none(
        self, mock_get_config, mock_bridge_handler
    ):
        """When the provider has no native responses API config (returns None),
        the bridge should be used regardless of the flag (existing behavior)."""
        mock_get_config.return_value = None
        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="anthropic/claude-3-haiku",
            input="Hello",
            litellm_logging_obj=MagicMock(),
        )

        mock_bridge_handler.assert_called_once()

    @patch("litellm.acompletion")
    @patch.object(
        import_module("litellm.responses.main").ProviderConfigManager, "get_provider_responses_api_config"
    )
    async def test_allowed_openai_params_forwarded_through_bridge(
        self, mock_get_config, mock_acompletion
    ):
        """allowed_openai_params is a named param of responses(), so it must be
        explicitly forwarded to the bridge; otherwise litellm.acompletion raises
        UnsupportedParamsError for params the caller explicitly allowed."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_acompletion.return_value = ModelResponse(
            id="chatcmpl_123",
            model="openai/my-custom-model",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content="Answer"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        await litellm.aresponses(
            model="openai/my-custom-model",
            input="Hello",
            use_chat_completions_api=True,
            allowed_openai_params=["reasoning_effort"],
            reasoning={"effort": "high"},
            litellm_logging_obj=MagicMock(),
        )

        mock_acompletion.assert_called_once()
        assert mock_acompletion.call_args.kwargs.get("allowed_openai_params") == [
            "reasoning_effort"
        ]

    @pytest.mark.parametrize(
        ("model", "upstream_url", "use_chat_completions_api", "allowed_openai_params", "expected_chat_template_kwargs"),
        [
            pytest.param(
                "openai/my-custom-model",
                "https://api.openai.com/v1/chat/completions",
                True,
                None,
                None,
                id="native-config-drops-unknown-param",
            ),
            pytest.param(
                "openai/my-custom-model",
                "https://api.openai.com/v1/chat/completions",
                True,
                ["chat_template_kwargs"],
                {"thinking": True},
                id="native-config-keeps-allowed-param",
            ),
            pytest.param(
                "together_ai/my-custom-model",
                "https://api.together.ai/v1/chat/completions",
                False,
                None,
                {"thinking": True},
                id="no-native-config-keeps-passthrough",
            ),
        ],
    )
    def test_bridge_forwards_same_params_as_native_dispatch(
        self,
        model: str,
        upstream_url: str,
        use_chat_completions_api: bool,
        allowed_openai_params: list[str] | None,
        expected_chat_template_kwargs: dict[str, bool] | None,
        respx_mock: respx.MockRouter,
    ):
        upstream: Final = respx_mock.post(upstream_url).mock(
            return_value=httpx.Response(
                status_code=200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": "my-custom-model",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
                },
            )
        )

        response: Final = litellm.responses(
            model=model,
            input="Hello",
            use_chat_completions_api=use_chat_completions_api,
            allowed_openai_params=allowed_openai_params,
            chat_template_kwargs={"thinking": True},
            drop_params=True,
            api_key="fake-provider-api-key",
            num_retries=0,
        )

        assert upstream.call_count == 1
        request_body: Final = json.loads(upstream.calls[0].request.read())
        assert request_body.get("chat_template_kwargs") == expected_chat_template_kwargs
        assert request_body["messages"] == [{"role": "user", "content": "Hello"}]
        assert response.output[0].content[0].text == "Answer"

    def test_bridge_drops_client_metadata_even_when_allowed_openai_params_names_it(
        self, respx_mock: respx.MockRouter
    ):
        upstream: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                status_code=200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": "my-custom-model",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
                },
            )
        )

        response: Final = litellm.responses(
            model="openai/my-custom-model",
            input="Hello",
            use_chat_completions_api=True,
            allowed_openai_params=["client_metadata"],
            client_metadata={"turn_id": "turn-1", "thread_id": "thread-1"},
            api_key="fake-provider-api-key",
            num_retries=0,
        )

        assert upstream.call_count == 1
        request_body: Final = json.loads(upstream.calls[0].request.read())
        assert "client_metadata" not in request_body
        assert request_body["messages"] == [{"role": "user", "content": "Hello"}]
        assert response.output[0].content[0].text == "Answer"

    def test_bridge_merges_instructions_and_developer_input_for_databricks(self, respx_mock: respx.MockRouter):
        upstream: Final = respx_mock.post("https://example.databricks.test/serving-endpoints/chat/completions").mock(
            return_value=httpx.Response(
                status_code=200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": "my-custom-model",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
                },
            )
        )

        response: Final = litellm.responses(
            model="databricks/my-custom-model",
            instructions="You are terse.",
            input=[
                {"role": "developer", "content": [{"type": "input_text", "text": "Skills: none."}]},
                {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
            ],
            client_metadata={"turn_id": "turn-1", "thread_id": "thread-1"},
            chat_template_kwargs={"thinking": True},
            api_base="https://example.databricks.test/serving-endpoints",
            api_key="fake-databricks-api-key",
            num_retries=0,
        )

        assert upstream.call_count == 1
        request_body: Final = json.loads(upstream.calls[0].request.read())
        assert request_body["messages"] == [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are terse."}, {"type": "text", "text": "Skills: none."}],
            },
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        assert "client_metadata" not in request_body
        assert request_body["chat_template_kwargs"] == {"thinking": True}
        assert response.output[0].content[0].text == "Answer"

    def test_bridge_drops_client_metadata_for_provider_without_native_config(self, respx_mock: respx.MockRouter):
        upstream: Final = respx_mock.post("https://example.databricks.test/serving-endpoints/chat/completions").mock(
            return_value=httpx.Response(
                status_code=200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": "my-custom-model",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
                },
            )
        )

        response: Final = litellm.responses(
            model="databricks/my-custom-model",
            input="Hello",
            client_metadata={
                "turn_id": "turn-1",
                "thread_id": "thread-1",
                "session_id": "session-1",
                "root_turn_id": "turn-1",
                "x-codex-installation-id": "install-1",
                "x-codex-turn-metadata": '{"turn_id":"turn-1"}',
            },
            chat_template_kwargs={"thinking": True},
            api_base="https://example.databricks.test/serving-endpoints",
            api_key="fake-databricks-api-key",
            num_retries=0,
        )

        assert upstream.call_count == 1
        request_body: Final = json.loads(upstream.calls[0].request.read())
        assert "client_metadata" not in request_body
        assert request_body["chat_template_kwargs"] == {"thinking": True}
        assert request_body["messages"] == [{"role": "user", "content": "Hello"}]
        assert response.output[0].content[0].text == "Answer"

    def test_bridge_keeps_deployment_credentials_while_dropping_unknown_params(self, respx_mock: respx.MockRouter):
        upstream: Final = respx_mock.post(
            "https://example-resource.openai.azure.com/openai/deployments/my-deployment/chat/completions",
            params={"api-version": "2024-10-21"},
        ).mock(
            return_value=httpx.Response(
                status_code=200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": "my-deployment",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
                },
            )
        )

        litellm.responses(
            model="azure/my-deployment",
            input="Hello",
            use_chat_completions_api=True,
            api_base="https://example-resource.openai.azure.com",
            api_version="2024-10-21",
            azure_ad_token="fake-azure-ad-token",
            chat_template_kwargs={"thinking": True},
            num_retries=0,
        )

        assert upstream.call_count == 1
        request: Final = upstream.calls[0].request
        assert request.headers["authorization"] == "Bearer fake-azure-ad-token"
        assert "chat_template_kwargs" not in json.loads(request.read())

    @patch.object(import_module("litellm.responses.file_search.emulated_handler"), "_call_aresponses")
    @patch.object(
        import_module("litellm.responses.main").ProviderConfigManager, "get_provider_responses_api_config"
    )
    async def test_bridge_flag_forwarded_to_file_search_emulation(
        self, mock_get_config, mock_call_aresponses
    ):
        """When use_chat_completions_api=True and file_search tool is present,
        the flag should be forwarded to the inner aresponses call in the
        file_search emulation path."""
        # Setup: provider has native responses API support
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()

        # Mock the inner aresponses call to return a valid response
        mock_response = ResponsesAPIResponse(
            id="resp_123",
            model="openai/my-custom-model",
            created_at=1234567890,
            output=[
                {"type": "message", "content": [{"type": "text", "text": "Answer"}]}
            ],
            usage=ResponseAPIUsage(
                input_tokens=10, output_tokens=5, total_tokens=15
            ),
        )
        mock_call_aresponses.return_value = mock_response

        await litellm.aresponses(
            model="openai/my-custom-model",
            input="Search for information",
            tools=[{"type": "file_search"}],
            use_chat_completions_api=True,
            litellm_logging_obj=MagicMock(),
        )

        # Verify _call_aresponses was called with use_chat_completions_api=True
        mock_call_aresponses.assert_called_once()
        call_kwargs = mock_call_aresponses.call_args.kwargs
        assert (
            call_kwargs.get("use_chat_completions_api") is True
        ), "use_chat_completions_api should be forwarded to inner aresponses call"

    @patch.object(
        import_module("litellm.responses.main").litellm_completion_transformation_handler, "response_api_handler"
    )
    @patch("litellm.vector_stores.main.asearch")
    @patch.object(
        import_module("litellm.responses.main").ProviderConfigManager, "get_provider_responses_api_config"
    )
    async def test_bridge_flag_prevents_native_responses_endpoint_call(
        self, mock_get_config, mock_asearch, mock_bridge_handler
    ):
        """
        Concrete failing scenario: native OpenAI responses config + bridge flag +
        file_search → emulation must still route inner calls through the bridge
        (chat completions), not POST to api_base /v1/responses.
        """
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_asearch.return_value = []

        first_response = ResponsesAPIResponse(
            id="resp_first",
            model="openai/my-local-model",
            created_at=1234567890,
            output=[
                {
                    "type": "function_call",
                    "name": "litellm_file_search",
                    "call_id": "call_123",
                    "arguments": '{"queries": ["test query"]}',
                }
            ],
            usage=ResponseAPIUsage(
                input_tokens=10, output_tokens=5, total_tokens=15
            ),
        )
        second_response = ResponsesAPIResponse(
            id="resp_second",
            model="openai/my-local-model",
            created_at=1234567891,
            output=[
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "Final answer"}],
                }
            ],
            usage=ResponseAPIUsage(
                input_tokens=20, output_tokens=10, total_tokens=30
            ),
        )
        mock_bridge_handler.side_effect = [first_response, second_response]

        result = await litellm.aresponses(
            model="openai/my-local-model",
            input="Search for information",
            tools=[
                {
                    "type": "file_search",
                    "file_search": {"vector_store_ids": ["vs_123"]},
                }
            ],
            use_chat_completions_api=True,
            api_base="http://localhost:8080/v1",
            litellm_logging_obj=MagicMock(),
        )

        assert mock_bridge_handler.call_count == 2, (
            "Bridge handler should be called twice: initial function-tool call "
            "and follow-up with tool results"
        )
        for call in mock_bridge_handler.call_args_list:
            all_kwargs = call.kwargs if call.kwargs else {}
            assert "use_chat_completions_api" not in all_kwargs
        assert result is not None
        assert result.id is not None

    @patch.object(import_module("litellm.responses.main").base_llm_http_handler, "response_api_handler")
    @patch("litellm.vector_stores.main.asearch")
    @patch.object(
        import_module("litellm.responses.main").ProviderConfigManager, "get_provider_responses_api_config"
    )
    async def test_without_bridge_flag_uses_native_endpoint(
        self, mock_get_config, mock_asearch, mock_native_handler
    ):
        """Without the bridge flag, openai/ with native config uses the native handler."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_asearch.return_value = []
        mock_native_handler.return_value = ResponsesAPIResponse(
            id="resp_native",
            model="openai/gpt-4o",
            created_at=1234567890,
            output=[
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "Native response"}],
                }
            ],
            usage=ResponseAPIUsage(
                input_tokens=10, output_tokens=5, total_tokens=15
            ),
        )

        result = await litellm.aresponses(
            model="openai/gpt-4o",
            input="Hello",
            litellm_logging_obj=MagicMock(),
        )

        mock_native_handler.assert_called_once()
        assert result is not None


def test_hosted_bridge_preserves_wire_history_custom_identity_and_request_metadata() -> None:
    from litellm.responses.litellm_completion_transformation.custom_tools import native_responses_custom_tool_name_map

    tools: Final = [
        {
            "type": "namespace",
            "name": ns,
            "description": f"Shared instructions for {ns}.",
            "tools": [{"type": "custom", "name": "exec", "description": "Execute fixture code."}],
        }
        for ns in ("alpha", "beta")
    ] + [{"type": "function", "name": "alpha__exec", "parameters": {"type": "object"}, "strict": True}]
    wire_name: Final = next(
        name
        for name, identity in native_responses_custom_tool_name_map({"tools": tools}).items()
        if identity == ("exec", "alpha")
    )

    def respond(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["messages"][0]["content"].count("Shared instructions for alpha.") == 1
        assert body["messages"][1]["role"] == "system"
        assert body["messages"][2]["reasoning_content"] == "  Think\ncarefully  "
        assert body["messages"][2]["tool_calls"][0]["function"]["name"] == wire_name
        assert body["messages"][3]["tool_call_id"] == "call_previous"
        assert body["messages"][3]["content"] == "first\nsecond"
        assert body["reasoning_effort"] == "medium"
        assert body["tool_choice"] == "required"
        assert [tool["function"]["name"] for tool in body["tools"]] == ["beta__exec"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "Next thought",
                            "tool_calls": [
                                {
                                    "id": "call_next",
                                    "type": "function",
                                    "function": {
                                        "name": "beta__exec",
                                        "arguments": json.dumps({"content": "fixture()"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result: Final = litellm.responses(
            model="hosted_vllm/fixture",
            api_base="https://fixture.invalid/v1",
            api_key="fixture",
            client=HTTPHandler(client=client),
            use_chat_completions_api=True,
            num_retries=0,
            input=[
                {"role": "developer", "content": "Preserve developer content."},
                {
                    "type": "reasoning",
                    "content": [
                        {"type": "reasoning_text", "text": "  Think\n"},
                        {"type": "reasoning_text", "text": "carefully  "},
                    ],
                },
                {
                    "type": "custom_tool_call",
                    "name": "exec",
                    "namespace": "alpha",
                    "call_id": "call_previous",
                    "input": "previous()",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_previous",
                    "output": [{"type": "input_text", "text": "first\n"}, {"type": "input_text", "text": "second"}],
                },
                {"role": "user", "content": "Continue"},
            ],
            tools=tools,
            parallel_tool_calls=True,
            metadata={"fixture": "correlation"},
            reasoning={"effort": "medium"},
            tool_choice={
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "custom", "name": "exec", "namespace": "beta"}],
            },
        )
    assert result.metadata == {"fixture": "correlation"}
    assert result.model_dump(mode="json", exclude_none=True)["tools"] == tools
    assert result.reasoning == {"effort": "medium"}
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.parallel_tool_calls is True
    assert result.output[0].content[0].type == "reasoning_text"
    assert result.output[0].content[0].text == "Next thought"
    assert result.output[1].type == "custom_tool_call"
    assert result.output[1].name == "exec"
    assert result.output[1].namespace == "beta"
    assert result.output[1].call_id == "call_next"
    assert result.output[1].input == "fixture()"


@pytest.mark.parametrize(
    "item, message",
    (
        ({"type": "reasoning", "encrypted_content": "opaque_fixture"}, "encrypted reasoning"),
        (
            {
                "type": "reasoning",
                "encrypted_content": "opaque_fixture",
                "content": [{"type": "reasoning_text", "text": "visible"}],
            },
            "encrypted reasoning",
        ),
        (
            {
                "type": "function_call_output",
                "call_id": "call_fixture",
                "output": [
                    {"type": "input_text", "text": "before"},
                    {"type": "input_image", "image_url": "data:image/png;base64,fixture"},
                    {"type": "input_text", "text": "after"},
                ],
            },
            "images in tool results",
        ),
        (
            {
                "type": "function_call_output",
                "call_id": "call_fixture",
                "output": [{"type": "input_text", "text": "keep"}, {"type": "input_file", "file_id": "fixture"}],
            },
            "only text parts",
        ),
        (
            {"type": "function_call_output", "call_id": "call_fixture", "output": ["untyped part"]},
            "must be text objects",
        ),
        ({"type": "function_call_output", "output": "unpaired"}, "nonempty call_id"),
    ),
)
def test_hosted_bridge_rejects_unsupported_history_before_sending(item, message) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        pytest.fail("Unsupported history reached the backend")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(litellm.BadRequestError, match=message) as error:
            litellm.responses(
                model="hosted_vllm/fixture",
                api_base="https://fixture.invalid/v1",
                api_key="fixture",
                client=HTTPHandler(client=client),
                use_chat_completions_api=True,
                num_retries=0,
                input=[item, {"role": "user", "content": "Continue"}],
            )
    assert error.value.status_code == 400
