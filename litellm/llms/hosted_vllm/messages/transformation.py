from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm.litellm_core_utils.reasoning_effort_utils import reasoning_effort_from_thinking_budget
from litellm.llms.openai_like.messages.transformation import (
    OpenAILikeAnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams

from ..reasoning import ReasoningEffortConfig, get_reasoning_effort_config

_JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])
_EMPTY_JSON_OBJECT: Final[Mapping[str, object]] = MappingProxyType({})


def _normalize_messages_reasoning(
    reasoning_config: ReasoningEffortConfig,
    reasoning_effort: object,
    thinking: object,
    output_effort: object,
    model: str,
) -> tuple[Literal["enabled", "disabled"] | None, str | None]:
    thinking_mapping: Final = (
        _JSON_OBJECT_ADAPTER.validate_python(thinking) if isinstance(thinking, Mapping) else _EMPTY_JSON_OBJECT
    )
    thinking_type: Final = thinking_mapping.get("type")
    if thinking_type == "disabled":
        reasoning_config.normalize("none", model=model)
        return "disabled", None

    budget_tokens: Final = thinking_mapping.get("budget_tokens", 0)
    budget_effort: Final = (
        reasoning_effort_from_thinking_budget(budget_tokens)
        if thinking_type == "enabled" and isinstance(budget_tokens, int)
        else None
    )
    requested_effort: Final = (
        output_effort
        if output_effort is not None
        else reasoning_effort
        if reasoning_effort is not None
        else budget_effort
    )
    normalized_effort: Final = reasoning_config.normalize(requested_effort, model=model)
    if normalized_effort == "disabled":
        return "disabled", None
    mode: Final = "enabled" if thinking_type in ("adaptive", "enabled") else None
    return mode, normalized_effort


class HostedVLLMAnthropicMessagesConfig(OpenAILikeAnthropicMessagesConfig):
    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict[str, object]],  # mutable-ok: inherited Anthropic adapter contract
        anthropic_messages_optional_request_params: dict[str, object],  # mutable-ok: inherited adapter request contract
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, object],  # mutable-ok: inherited HTTP header contract
    ) -> dict[str, object]:  # mutable-ok: inherited provider request contract
        raw_model_info: Final[object] = litellm_params.model_info
        model_info: Final = (
            _JSON_OBJECT_ADAPTER.validate_python(raw_model_info) if raw_model_info is not None else _EMPTY_JSON_OBJECT
        )
        reasoning_config: Final = get_reasoning_effort_config(model_info.get("reasoning_effort"))
        raw_model_extra: Final[object] = litellm_params.model_extra
        model_extra: Final = _JSON_OBJECT_ADAPTER.validate_python(raw_model_extra or _EMPTY_JSON_OBJECT)
        raw_template_kwargs: Final = model_extra.get("chat_template_kwargs")
        explicit_template_kwargs: Final = (
            _JSON_OBJECT_ADAPTER.validate_python(raw_template_kwargs)
            if isinstance(raw_template_kwargs, Mapping)
            else None
        )
        raw_output_config: Final = anthropic_messages_optional_request_params.get("output_config")
        output_config: Final = (
            _JSON_OBJECT_ADAPTER.validate_python(raw_output_config) if isinstance(raw_output_config, Mapping) else None
        )
        output_effort: Final = output_config.get("effort") if output_config is not None else None
        residual_output_config: Final = (
            {  # mutable-ok: output_config must remain a JSON object for the provider request
                key: value for key, value in output_config.items() if key != "effort"
            }
            if output_config is not None
            else None
        )
        request_params: Final = (
            {  # mutable-ok: inherited adapter consumes a mutable request-parameter mapping
                **{  # mutable-ok: filtering creates the provider request parameter JSON object
                    key: value
                    for key, value in anthropic_messages_optional_request_params.items()
                    if key not in ("reasoning_effort", "thinking", "output_config")
                },
                **(
                    {"output_config": residual_output_config}  # mutable-ok: conditional JSON request fragment
                    if residual_output_config
                    else {}  # mutable-ok: conditional JSON request fragment
                ),
            }
            if reasoning_config is not None
            else anthropic_messages_optional_request_params
        )
        reasoning_mode, effort = (
            _normalize_messages_reasoning(
                reasoning_config=reasoning_config,
                reasoning_effort=anthropic_messages_optional_request_params.get("reasoning_effort"),
                thinking=anthropic_messages_optional_request_params.get("thinking"),
                output_effort=output_effort,
                model=model,
            )
            if reasoning_config is not None
            else (None, None)
        )

        raw_base_request: Final[object] = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        base_request: Final = _JSON_OBJECT_ADAPTER.validate_python(raw_base_request)
        request: Final = (
            {  # mutable-ok: provider adapter contract requires a mutable request object
                **base_request,
                "chat_template_kwargs": explicit_template_kwargs.copy(),
            }
            if explicit_template_kwargs is not None
            else base_request
        )
        if reasoning_config is None or explicit_template_kwargs is not None:
            return request
        template_kwargs: Final = (
            reasoning_config.get_chat_template_kwargs("disabled", model=model)
            if reasoning_mode == "disabled"
            else reasoning_config.get_chat_template_kwargs(effort, model=model)
            if reasoning_config.target == "chat_template_kwargs" and effort is not None
            else reasoning_config.enabled
            if reasoning_mode == "enabled"
            else _EMPTY_JSON_OBJECT
        )
        normalized_request: Final = (
            {  # mutable-ok: provider adapter contract requires a mutable request object
                **request,
                "chat_template_kwargs": dict(  # mutable-ok: vLLM consumes a mutable JSON object
                    template_kwargs
                ),
            }
            if template_kwargs
            else request
        )
        if reasoning_config.target != "native" or effort is None or reasoning_mode == "disabled":
            return normalized_request

        request_output_config: Final = normalized_request.get("output_config")
        existing_output_config: Final = (
            _JSON_OBJECT_ADAPTER.validate_python(request_output_config)
            if isinstance(request_output_config, Mapping)
            else {}  # mutable-ok: output_config is emitted as a JSON object
        )
        return {  # mutable-ok: provider adapter contract requires a mutable request object
            **normalized_request,
            "output_config": {  # mutable-ok: output_config is emitted as a JSON object
                **existing_output_config,
                "effort": effort,
            },
        }
