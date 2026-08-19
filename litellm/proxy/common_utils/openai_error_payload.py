"""Shapes the ``error`` object the proxy answers with so it matches OpenAI's contract:
``type`` is a required string and ``param`` is nullable, neither of which the literal
string ``"None"`` satisfies."""

import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from uuid import uuid4

from fastapi import status

from litellm.constants import STRINGIFIED_NONE
from litellm.proxy._types import ProxyException

LITELLM_CALL_ID_HEADER: Final = "x-litellm-call-id"
from litellm._logging import redact_internal_details_from_client_message
from litellm.exceptions import ContextWindowExceededError
from litellm.litellm_core_utils.exception_mapping_utils import extract_error_message_from_string
from litellm.router_utils.add_retry_fallback_headers import HiddenParamsAsyncIteratorWrapper
from litellm.types.llms.openai import ResponsesAPIResponse

_OPENAI_ERROR_TYPE_BY_STATUS: Final[Mapping[int, str]] = MappingProxyType(
    {
        status.HTTP_401_UNAUTHORIZED: "authentication_error",
        status.HTTP_403_FORBIDDEN: "permission_error",
        status.HTTP_429_TOO_MANY_REQUESTS: "rate_limit_error",
    }
)


def attribute_of(value: object, name: str, default: object = None) -> object:
    return getattr(value, name, default)


def error_status_code(exc: object, default: int) -> int:
    """The HTTP status an exception carries as ``status_code`` or, the way ``ProxyException``
    stores it, as a stringified ``code``; ``default`` when it carries neither."""
    carried: Final = attribute_of(exc, "status_code")
    if isinstance(carried, int) and not isinstance(carried, bool):
        return carried
    stringified: Final = attribute_of(exc, "code")
    return int(stringified) if isinstance(stringified, str) and stringified.isdecimal() else default


def openai_error_type(exc: object, status_code: int) -> str:
    """OpenAI types ``error.type`` as a required string, so an exception carrying none
    falls back to the type its status code stands for."""
    carried: Final = attribute_of(exc, "type")
    if isinstance(carried, str) and carried != STRINGIFIED_NONE:
        return carried
    mapped: Final = _OPENAI_ERROR_TYPE_BY_STATUS.get(status_code)
    if mapped is not None:
        return mapped
    if status_code < status.HTTP_500_INTERNAL_SERVER_ERROR:
        return "invalid_request_error"
    return "internal_server_error"


def openai_error_param(exc: object) -> str | None:
    """OpenAI types ``error.param`` as nullable, so an exception carrying none
    serializes as JSON ``null``."""
    carried: Final = attribute_of(exc, "param")
    return carried if isinstance(carried, str) and carried != STRINGIFIED_NONE else None


def litellm_call_id_headers(litellm_call_id: str | None) -> dict[str, str] | None:  # mutable-ok: ProxyException.headers
    if litellm_call_id is None:
        return None
    return {LITELLM_CALL_ID_HEADER: litellm_call_id}  # mutable-ok: ProxyException mutates its headers dict


def with_litellm_call_id(exc: ProxyException, litellm_call_id: str | None) -> ProxyException:
    """The same error object, answering with ``x-litellm-call-id`` when it was raised without one."""
    if litellm_call_id is not None:
        exc.headers.setdefault(LITELLM_CALL_ID_HEADER, litellm_call_id)
    return exc


def headers_with_litellm_call_id(headers: Mapping[str, str] | None, litellm_call_id: str) -> Mapping[str, str]:
    """``headers`` plus ``x-litellm-call-id``, keeping the value they already carry under that name."""
    if headers is None:
        return MappingProxyType({LITELLM_CALL_ID_HEADER: litellm_call_id})
    return MappingProxyType({LITELLM_CALL_ID_HEADER: litellm_call_id, **headers})


def is_context_window_error(exc: object) -> bool:
    return isinstance(exc, ContextWindowExceededError) or any(
        attribute_of(exc, field) == "context_length_exceeded" for field in ("code", "openai_code")
    )


class ResponsesContextErrorFormatter:
    def __init__(self, model: object) -> None:
        self._model = model if isinstance(model, str) else None
        self._response: ResponsesAPIResponse | None = None
        self._sequence_number = -1

    def observe(self, chunk: object) -> None:
        sequence: Final = attribute_of(chunk, "sequence_number")
        self._sequence_number = (
            max(self._sequence_number + 1, sequence) if isinstance(sequence, int) else self._sequence_number + 1
        )
        response: Final = attribute_of(chunk, "response")
        if isinstance(response, ResponsesAPIResponse):
            self._response = response

    def format(self, exc: Exception, *, stream: object = None) -> str | None:
        if not is_context_window_error(exc):
            return None
        raw_message: Final = attribute_of(exc, "message", str(exc))
        message: Final = raw_message if isinstance(raw_message, str) else str(exc)
        safe_message: Final = redact_internal_details_from_client_message(
            extract_error_message_from_string(message) or message
        )
        source: Final = (
            attribute_of(stream, "_inner") if isinstance(stream, HiddenParamsAsyncIteratorWrapper) else stream
        )
        terminal: Final = attribute_of(source, "completed_response")
        terminal_response: Final = attribute_of(terminal, "response")
        response: Final = (
            terminal_response if isinstance(terminal_response, ResponsesAPIResponse) else self._response
        ) or ResponsesAPIResponse.model_validate(
            MappingProxyType(
                {
                    "id": f"resp_{uuid4().hex}",
                    "object": "response",
                    "created_at": int(time.time()),
                    "model": self._model,
                    "output": (),
                }
            )
        )
        failed: Final = ResponsesAPIResponse.model_validate(
            MappingProxyType(
                {
                    **response.model_dump(),
                    "status": "failed",
                    "error": MappingProxyType({"code": "context_length_exceeded", "message": safe_message}),
                }
            )
        )
        from litellm.types.llms.openai import ResponseFailedEvent, ResponsesAPIStreamEvents

        event: Final = ResponseFailedEvent(type=ResponsesAPIStreamEvents.RESPONSE_FAILED, response=failed)
        terminal_sequence: Final = attribute_of(terminal, "sequence_number")
        sequence: Final = (
            max(self._sequence_number + 1, terminal_sequence)
            if isinstance(terminal_sequence, int)
            else self._sequence_number + 1
        )
        payload: Final = event.model_copy(update=MappingProxyType({"sequence_number": sequence})).model_dump_json()
        return f"event: response.failed\ndata: {payload}\n\n"
