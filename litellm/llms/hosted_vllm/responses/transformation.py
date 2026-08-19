"""
Responses API transformation for Hosted VLLM provider.

vLLM natively supports the OpenAI-compatible /v1/responses endpoint,
so this config enables direct routing instead of falling back to
the chat completions → responses conversion pipeline.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import TypeAdapter

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.litellm_completion_transformation.custom_tools import (
    native_responses_custom_tool_name_map,
    native_responses_namespace_tool_name_map,
    normalize_native_responses_custom_tools,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..reasoning import get_reasoning_effort_config
from .custom_tools import NativeResponsesCustomToolAdapter

_JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])
_INPUT_ITEMS_ADAPTER: Final = TypeAdapter(tuple[Mapping[str, object], ...])
_EMPTY_JSON_OBJECT: Final[Mapping[str, object]] = MappingProxyType({})


def _with_developer_messages_as_system(request: Mapping[str, object]) -> Mapping[str, object]:
    input: Final = request.get("input")
    if isinstance(input, str):
        return request
    items: Final = _INPUT_ITEMS_ADAPTER.validate_python(input)
    return MappingProxyType(
        {
            **request,
            "input": tuple(
                _JSON_OBJECT_ADAPTER.validate_python(MappingProxyType({**item, "role": "system"}))
                if item.get("type", "message") == "message" and item.get("role") == "developer"
                else item
                for item in items
            ),
        }
    )


class HostedVLLMResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for Hosted VLLM Responses API support.

    Extends OpenAI's config since vLLM follows OpenAI's API spec,
    but uses HOSTED_VLLM_API_BASE for the base URL and defaults
    to "fake-api-key" when no API key is provided (vLLM does not
    require authentication by default).
    """

    def __init__(self) -> None:
        super().__init__()
        self._custom_tools: NativeResponsesCustomToolAdapter | None = None

    def should_fake_stream(
        self, model: str | None, stream: bool | None, custom_llm_provider: str | None = None
    ) -> bool:
        return False

    def prepare_streaming_chunk(self, chunk: str) -> tuple[str, ...]:
        return self._custom_tools.chunk(chunk) if self._custom_tools is not None else (chunk,)

    def transform_response_api_response(
        self, model: str, raw_response: httpx.Response, logging_obj: Logging
    ) -> ResponsesAPIResponse:
        response: Final = super().transform_response_api_response(model, raw_response, logging_obj)
        if self._custom_tools is None:
            return response
        restored: Final = ResponsesAPIResponse.model_validate(self._custom_tools.response(response.model_dump()))
        return response.model_copy(update=vars(restored))

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.HOSTED_VLLM

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key: Final = (
            litellm_params.api_key or get_secret_str("HOSTED_VLLM_API_KEY") or "fake-api-key"
        )  # vllm does not require an api key
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, object],  # mutable-ok: inherited HTTP header contract
    ) -> dict[str, object]:  # mutable-ok: inherited provider request contract
        raw_request: Final[object] = super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        original_request: Final = _JSON_OBJECT_ADAPTER.validate_python(raw_request)
        custom_names: Final = native_responses_custom_tool_name_map(original_request)
        namespace_names: Final = native_responses_namespace_tool_name_map(original_request)
        self._custom_tools = (
            NativeResponsesCustomToolAdapter(custom_names, namespace_names) if custom_names or namespace_names else None
        )
        raw_model_info: Final[object] = litellm_params.get("model_info")
        model_info: Final = (
            _JSON_OBJECT_ADAPTER.validate_python(raw_model_info)
            if isinstance(raw_model_info, Mapping)
            else _EMPTY_JSON_OBJECT
        )
        role_safe_request: Final = (
            _with_developer_messages_as_system(original_request)
            if model_info.get("supports_developer_messages") is False
            else original_request
        )
        request: Final = _JSON_OBJECT_ADAPTER.validate_python(
            normalize_native_responses_custom_tools(role_safe_request)
        )
        reasoning_config: Final = get_reasoning_effort_config(model_info.get("reasoning_effort"))
        if reasoning_config is None:
            return request

        raw_reasoning: Final = request.get("reasoning")
        reasoning: Final = (
            _JSON_OBJECT_ADAPTER.validate_python(raw_reasoning) if isinstance(raw_reasoning, Mapping) else None
        )
        raw_effort: Final = reasoning.get("effort") if reasoning is not None else None
        effort: Final = reasoning_config.normalize(raw_effort, model=model)
        request["reasoning"] = {  # mutable-ok: request payload must be a mapping
            **(reasoning or _EMPTY_JSON_OBJECT),
            "effort": "none" if effort == "disabled" else effort,
        }
        return request

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        api_base = api_base or get_secret_str("HOSTED_VLLM_API_BASE")

        if api_base is None:
            raise ValueError(
                "api_base not set for Hosted VLLM responses API. "
                "Set via api_base parameter or HOSTED_VLLM_API_BASE environment variable"
            )

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        # If api_base already ends with /v1, append /responses
        # Otherwise append /v1/responses
        if api_base.endswith("/v1"):
            return f"{api_base}/responses"

        return f"{api_base}/v1/responses"

    def supports_native_websocket(self) -> bool:
        """Hosted vLLM does not support native WebSocket for Responses API"""
        return False
