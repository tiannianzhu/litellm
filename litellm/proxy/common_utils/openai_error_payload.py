"""Shapes the ``error`` object the proxy answers with so it matches OpenAI's contract:
``type`` is a required string and ``param`` is nullable, neither of which the literal
string ``"None"`` satisfies."""

import asyncio
import re
import socket
import ssl
import time
from collections.abc import Iterator, Mapping
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit
from uuid import uuid4

import aiohttp
import httpx
from fastapi import Request, status
from pydantic import TypeAdapter

from litellm.constants import STRINGIFIED_NONE
from litellm.proxy._types import ProxyException

LITELLM_CALL_ID_HEADER: Final = "x-litellm-call-id"
from litellm._logging import redact_internal_details_from_client_message
from litellm.exceptions import LITELLM_EXCEPTION_TYPES, ContextWindowExceededError, MidStreamFallbackError
from litellm.litellm_core_utils.bug_report import strip_bug_report_notice
from litellm.litellm_core_utils.exception_mapping_utils import (
    extract_error_message_from_dict,
    extract_error_message_from_string,
)
from litellm.router_utils.add_retry_fallback_headers import HiddenParamsAsyncIteratorWrapper
from litellm.types.llms.openai import ResponsesAPIResponse

_OPENAI_ERROR_TYPE_BY_STATUS: Final[Mapping[int, str]] = MappingProxyType(
    {
        status.HTTP_401_UNAUTHORIZED: "authentication_error",
        status.HTTP_403_FORBIDDEN: "permission_error",
        status.HTTP_429_TOO_MANY_REQUESTS: "rate_limit_error",
    }
)
_ERROR_MAPPING: Final = TypeAdapter(Mapping[str, object])
_EXCEPTION_NAMES: Final = frozenset(exception.__name__ for exception in LITELLM_EXCEPTION_TYPES)
_EXCEPTION_LABEL_ALIASES: Final = frozenset(
    {"Timeout Error", "AnthropicError", "APITimeoutError", "GetLLMProvider Exception"}
)
_PROVIDER_EXCEPTION_LABEL: Final = re.compile(r"[A-Za-z_][A-Za-z_0-9]*Exception")
_WRAPPER_SEPARATORS: Final = (": ", " - ")
_ROUTER_MESSAGE_SUFFIXES: Final = (
    "\n\nLiteLLM: model group '",
    "\n\nDeployment Info:",
)
_TIMEOUT_FAILURE_TYPES: Final = (httpx.TimeoutException, TimeoutError, asyncio.TimeoutError)
_TRANSPORT_FAILURE_TYPES: Final = (
    httpx.TransportError,
    aiohttp.ClientConnectionError,
    ConnectionError,
    socket.gaierror,
    ssl.SSLError,
)
_TIMEOUT_FAILURE_MESSAGE: Final = "The model service timed out."
_UNREACHABLE_FAILURE_MESSAGE: Final = "Could not reach the model service."


def inference_error_surface(path: str, method: str = "POST") -> Literal["chat", "responses", "messages"] | None:
    if method != "POST":
        return None
    normalized: Final = urlsplit(path).path.rstrip("/")
    if normalized in ("/responses", "/v1/responses", "/openai/v1/responses"):
        return "responses"
    if normalized in ("/v1/messages", "/anthropic/v1/messages"):
        return "messages"
    if normalized in ("/chat/completions", "/v1/chat/completions", "/cursor/chat/completions") or (
        normalized.endswith("/chat/completions") and normalized.startswith(("/engines/", "/openai/deployments/"))
    ):
        return "chat"
    return None


def inference_request_surface(request: Request) -> Literal["chat", "responses", "messages"] | None:
    scope: Final = _ERROR_MAPPING.validate_python(request.scope)
    route_path: Final = attribute_of(scope.get("route"), "path")
    root_path: Final = scope.get("root_path", "")
    return inference_error_surface(
        route_path
        if isinstance(route_path, str)
        else request.url.path.removeprefix(root_path if isinstance(root_path, str) else ""),
        request.method,
    )


def _message_from_value(value: object) -> str | None:
    if isinstance(value, str):
        return extract_error_message_from_string(value) or value or None
    if not isinstance(value, Mapping):
        return None
    mapping: Final = _ERROR_MAPPING.validate_python(value)
    return extract_error_message_from_dict(mapping) or next(
        (text for key in ("detail", "error") if isinstance(text := mapping.get(key), str)), None
    )


def _response_error_message(exc: object) -> str | None:
    response: Final = attribute_of(exc, "response")
    if not isinstance(response, httpx.Response):
        return None
    try:
        text: Final = response.text
    except httpx.ResponseNotRead:
        return None
    return (
        _message_from_value(text) if isinstance(exc, httpx.HTTPStatusError) else extract_error_message_from_string(text)
    )


def _linked_exceptions(node: object) -> Iterator[BaseException]:
    status_code: Final = attribute_of(node, "status_code")
    if isinstance(node, httpx.HTTPStatusError) or (
        isinstance(status_code, int)
        and status.HTTP_400_BAD_REQUEST <= status_code < status.HTTP_500_INTERNAL_SERVER_ERROR
        and status_code != status.HTTP_408_REQUEST_TIMEOUT
    ):
        return
    cause: Final = attribute_of(node, "__cause__")
    context: Final = None if attribute_of(node, "__suppress_context__", False) else attribute_of(node, "__context__")
    for linked in (attribute_of(node, "original_exception"), cause if cause is not None else context):
        if isinstance(linked, BaseException):
            yield linked


def _exception_chain(exc: object) -> tuple[object, ...]:
    def _expand(pending: tuple[object, ...], seen: frozenset[int]) -> tuple[object, ...]:
        fresh: Final = tuple({id(node): node for node in pending if id(node) not in seen}.values())
        if not fresh:
            return ()
        grown: Final = seen | frozenset(id(node) for node in fresh)
        linked: Final = tuple(chain.from_iterable(_linked_exceptions(node) for node in fresh))
        return fresh + _expand(linked, grown)

    return _expand((exc,), frozenset())


def _transport_failure_message(exc: object) -> str | None:
    for node in _exception_chain(exc):
        if isinstance(node, _TIMEOUT_FAILURE_TYPES):
            return _TIMEOUT_FAILURE_MESSAGE
        if isinstance(node, _TRANSPORT_FAILURE_TYPES):
            return _UNREACHABLE_FAILURE_MESSAGE
    return None


def _is_exception_label_token(token: str) -> bool:
    bare: Final = token.removeprefix("litellm.")
    return bare in _EXCEPTION_NAMES or _PROVIDER_EXCEPTION_LABEL.fullmatch(bare) is not None


def _is_internal_exception_label(segment: str) -> bool:
    tokens: Final = tuple(segment.strip().split())
    if " ".join(tokens) in _EXCEPTION_LABEL_ALIASES:
        return True
    return 0 < len(tokens) <= 2 and all(_is_exception_label_token(token) for token in tokens)


def _internal_wrapper_remainder(raw: str) -> str | None:
    for separator in _WRAPPER_SEPARATORS:
        prefix, matched, remainder = raw.partition(separator)
        if matched and _is_internal_exception_label(prefix):
            return remainder
    return None


def client_error_message(exc: object) -> str:
    if isinstance(exc, MidStreamFallbackError) and exc.original_exception is not None:
        return client_error_message(exc.original_exception)
    provider_message: Final = next(
        (
            message
            for value in (attribute_of(exc, "body"), attribute_of(exc, "detail"))
            if (message := _message_from_value(value)) is not None
        ),
        None,
    ) or _response_error_message(exc)
    transport_failure: Final = _transport_failure_message(exc) if provider_message is None else None
    if transport_failure is not None:
        return transport_failure
    source: Final = provider_message or _message_from_value(attribute_of(exc, "message")) or _message_from_value(exc)
    raw: Final = source if source is not None else str(exc)
    remainder: Final = _internal_wrapper_remainder(raw)
    unwrapped: Final = client_error_message(remainder) if remainder is not None else raw
    end: Final = min(
        (unwrapped.find(marker) for marker in _ROUTER_MESSAGE_SUFFIXES if marker in unwrapped), default=len(unwrapped)
    )
    return redact_internal_details_from_client_message(strip_bug_report_notice(unwrapped[:end].rstrip()))


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


def inference_error_status_code(exc: object) -> int:
    return exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else error_status_code(exc, 500)


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
    return {LITELLM_CALL_ID_HEADER: litellm_call_id}


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
    if isinstance(exc, MidStreamFallbackError) and exc.original_exception is not None:
        return is_context_window_error(exc.original_exception)
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
        safe_message: Final = client_error_message(exc)
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
