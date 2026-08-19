"""
Responses API transformation for Hosted VLLM provider.

vLLM natively supports the OpenAI-compatible /v1/responses endpoint,
so this config enables direct routing instead of falling back to
the chat completions → responses conversion pipeline.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..reasoning import get_reasoning_effort_config

_JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])
_EMPTY_JSON_OBJECT: Final[Mapping[str, object]] = MappingProxyType({})


class HostedVLLMResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for Hosted VLLM Responses API support.

    Extends OpenAI's config since vLLM follows OpenAI's API spec,
    but uses HOSTED_VLLM_API_BASE for the base URL and defaults
    to "fake-api-key" when no API key is provided (vLLM does not
    require authentication by default).
    """

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
        request: Final = _JSON_OBJECT_ADAPTER.validate_python(raw_request)
        raw_model_info: Final[object] = litellm_params.get("model_info")
        model_info: Final = (
            _JSON_OBJECT_ADAPTER.validate_python(raw_model_info)
            if isinstance(raw_model_info, Mapping)
            else _EMPTY_JSON_OBJECT
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
        if reasoning_config.target == "native":
            if effort is None:
                return request

            normalized_effort: Final = "none" if effort == "disabled" else effort
            request["reasoning"] = {  # mutable-ok: request payload must be a mapping
                **(reasoning or _EMPTY_JSON_OBJECT),
                "effort": normalized_effort,
            }
            return request

        request.pop("reasoning", None)
        if effort is None:
            return request

        configured_kwargs: Final = reasoning_config.get_chat_template_kwargs(effort, model=model)
        raw_existing_kwargs: Final = request.get("chat_template_kwargs")
        existing_kwargs: Final = (
            _JSON_OBJECT_ADAPTER.validate_python(raw_existing_kwargs)
            if isinstance(raw_existing_kwargs, Mapping)
            else _EMPTY_JSON_OBJECT
        )
        request["chat_template_kwargs"] = {  # mutable-ok: vLLM consumes template kwargs as a JSON object
            **configured_kwargs,
            **existing_kwargs,
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
