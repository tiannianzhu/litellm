import json
import re
from collections.abc import AsyncIterator, Mapping
from types import MappingProxyType
from typing import Final, Literal, cast

import httpx
from pydantic import TypeAdapter

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.reasoning_effort_utils import reasoning_effort_from_thinking_budget
from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import aclose_if_supported
from litellm.llms.openai_like.messages.transformation import (
    OpenAILikeAnthropicMessagesConfig,
)
from litellm.proxy.common_utils.sse_keepalive import split_complete_sse_frames
from litellm.types.llms.anthropic import AnthropicResponseContentBlockToolUse
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from litellm.types.router import GenericLiteLLMParams

from ..reasoning import ReasoningEffortConfig, get_reasoning_effort_config

_JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])
_EMPTY_JSON_OBJECT: Final[Mapping[str, object]] = MappingProxyType({})
_SSE_FRAME_DELIMITER: Final = re.compile(rb"\r\n\r\n|\n\n|\r\r")


def _correct_stop_reason_in_frame(frame: bytes, has_tool_use: bool) -> tuple[bytes, bool]:
    if b"content_block_start" not in frame and b"message_delta" not in frame:
        return frame, has_tool_use
    prefix, separator, data = frame.partition(b"data:")
    if not separator:
        return frame, has_tool_use
    payload_line: Final = data.splitlines(keepends=True)[0]
    suffix: Final = data[len(payload_line) :]
    line_ending: Final = payload_line[len(payload_line.rstrip(b"\r\n")) :]
    try:
        payload: Final = _JSON_OBJECT_ADAPTER.validate_json(payload_line)
    except ValueError:
        return frame, has_tool_use
    if payload.get("type") == "content_block_start":
        content_block: Final = payload.get("content_block")
        return frame, has_tool_use or (
            isinstance(content_block, Mapping)
            and _JSON_OBJECT_ADAPTER.validate_python(content_block).get("type") == "tool_use"
        )
    if payload.get("type") != "message_delta" or not has_tool_use:
        return frame, has_tool_use
    raw_delta: Final = payload.get("delta")
    if not isinstance(raw_delta, Mapping):
        return frame, has_tool_use
    delta: Final = _JSON_OBJECT_ADAPTER.validate_python(raw_delta)
    if delta.get("stop_reason") != "end_turn":
        return frame, has_tool_use
    corrected: Final[dict[str, object]] = {  # mutable-ok: JSON serialization requires a plain object
        **payload,
        "delta": {**delta, "stop_reason": "tool_use"},  # mutable-ok: one-shot SSE payload
    }
    return prefix + separator + json.dumps(corrected).encode() + line_ending + suffix, has_tool_use


async def _correct_vllm_messages_stream(stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    pending: bytes = b""  # rebind-ok: accumulates incomplete SSE frames across transport chunks
    has_tool_use: bool = False  # rebind-ok: records generated tool_use across the stream
    try:
        async for chunk in stream:
            complete, pending = split_complete_sse_frames(pending + chunk)
            if not complete:
                continue
            frame_start: int = 0
            for boundary in _SSE_FRAME_DELIMITER.finditer(complete):
                corrected, has_tool_use = _correct_stop_reason_in_frame(
                    complete[frame_start : boundary.end()], has_tool_use
                )
                yield corrected
                frame_start = boundary.end()
        if pending:
            yield pending
    finally:
        await aclose_if_supported(stream)


def _normalize_messages_reasoning(
    reasoning_config: ReasoningEffortConfig,
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
    requested_effort: Final = output_effort if output_effort is not None else budget_effort
    normalized_effort: Final = reasoning_config.normalize(requested_effort, model=model)
    if normalized_effort == "disabled":
        return "disabled", None
    mode: Final = "enabled" if thinking_type in ("adaptive", "enabled") else None
    return mode, normalized_effort


class HostedVLLMAnthropicMessagesConfig(OpenAILikeAnthropicMessagesConfig):
    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> AnthropicMessagesResponse:
        response: Final = super().transform_anthropic_messages_response(model, raw_response, logging_obj)
        content: Final = response.get("content")
        if (
            response.get("stop_reason") == "end_turn"
            and content
            and any(
                isinstance(block, AnthropicResponseContentBlockToolUse)
                or (
                    isinstance(block, Mapping) and _JSON_OBJECT_ADAPTER.validate_python(block).get("type") == "tool_use"
                )
                for block in content
            )
        ):
            return {**response, "stop_reason": "tool_use"}  # mutable-ok: inherited response is a TypedDict
        return response

    def get_async_streaming_response_iterator(
        self,
        model: str,
        httpx_response: httpx.Response,
        request_body: dict[str, object],  # mutable-ok: inherited streaming iterator contract
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> AsyncIterator[bytes]:
        stream: Final = cast(  # cast-ok: inherited chunk processor yields bytes, but its return is untyped
            AsyncIterator[bytes],
            super().get_async_streaming_response_iterator(  # pyright: ignore[reportUnknownMemberType]  # inherited return lacks bytes type
                model=model,
                httpx_response=httpx_response,
                request_body=request_body,
                litellm_logging_obj=litellm_logging_obj,
            ),
        )
        return _correct_vllm_messages_stream(stream)

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
                    if key not in ("thinking", "output_config")
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
        if reasoning_config is None:
            return base_request
        if reasoning_mode is not None:
            base_request["chat_template_kwargs"] = {  # mutable-ok: provider request must serialize as JSON
                "enable_thinking": reasoning_mode == "enabled"
            }
        if effort is not None:
            request_output_config: Final = base_request.get("output_config")
            existing_output_config: Final = (
                _JSON_OBJECT_ADAPTER.validate_python(request_output_config)
                if isinstance(request_output_config, Mapping)
                else _EMPTY_JSON_OBJECT
            )
            base_request["output_config"] = {  # mutable-ok: provider request must serialize as JSON
                **existing_output_config,
                "effort": effort,
            }
        return base_request
