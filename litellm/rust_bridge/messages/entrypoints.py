from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

import anyio

from litellm.rust_bridge import execution
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

_NATIVE_MESSAGES_LOGGING: Final[ContextVar[asyncio.Future[Logging] | None]] = ContextVar(
    "native_messages_logging", default=None
)


def register_messages_logging(logger: Logging) -> None:
    pending: Final = _NATIVE_MESSAGES_LOGGING.get()
    if pending is not None and not pending.done():
        pending.set_result(logger)


class MessagesStreamHead(dict[str, object]):
    agentic: Callable[[execution.AsyncStream], object] | None = None
    deliver: Callable[[object], Awaitable[object]] | None = None


@dataclass(frozen=True, slots=True)
class _Undelivered:
    stream: object
    deliver: Callable[[object], Awaitable[object]]


def _open_stream(native: execution.Execution, head: object, *, logging_future: asyncio.Future[Logging]) -> object:
    stream: Final = execution.AsyncStream(native, head)
    agentic: Final = head.agentic if isinstance(head, MessagesStreamHead) else None
    opened: Final = stream if agentic is None else agentic(stream)
    logger: Final = logging_future.result() if logging_future.done() else None
    accounted: Final = _accounted_stream(opened, stream, logger)
    deliver: Final = head.deliver if isinstance(head, MessagesStreamHead) else None
    return accounted if deliver is None else _Undelivered(accounted, deliver)


def _accounted_stream(opened: object, stream: execution.AsyncStream, logger: Logging | None) -> object:
    if logger is None or not logger.requires_success_await or not isinstance(opened, AsyncIterator):
        return opened
    from litellm.llms.anthropic.messages.native.streaming_iterator import (
        AnthropicMessagesStreamHiddenParams,
        AnthropicMessagesStreamingResponse,
    )

    return AnthropicMessagesStreamingResponse(
        completion_stream=cast("AsyncIterator[bytes]", opened),  # cast-ok: native Messages streams emit SSE bytes
        hidden_params=cast(  # cast-ok: native Messages stamps this shared header dictionary
            "AnthropicMessagesStreamHiddenParams",
            stream._hidden_params,  # pyright: ignore[reportPrivateUsage]  # native headers have no public getter
        ),
        accounting_logger=logger,
    )


async def _delivered(opened: Awaitable[object]) -> object:
    result: Final = await opened
    return await result.deliver(result.stream) if isinstance(result, _Undelivered) else result


class NativeMessages(Protocol):
    def __call__(
        self,
        request: NativeCall,
    ) -> AnthropicMessagesResponse | Iterator[bytes]: ...


class NativeAmessages(Protocol):
    def __call__(
        self,
        request: NativeCall,
    ) -> Awaitable[AnthropicMessagesResponse | AsyncIterator[bytes]]: ...


def _messages_binding(value: object) -> NativeMessages | None:
    if not callable(value):
        return None
    native: Final = execution.synchronous(value)
    return cast("NativeMessages", native)  # cast-ok: the execution adapter returns the route's result


def _amessages_binding(value: object) -> NativeAmessages | None:
    if not callable(value):
        return None

    async def amessages(request: NativeCall) -> object:
        from litellm.litellm_core_utils.litellm_logging import Logging

        logging_future: Final[asyncio.Future[Logging]] = asyncio.get_running_loop().create_future()
        supplied: Final = request.resolved.get("litellm_logging_obj")
        if isinstance(supplied, Logging):
            logging_future.set_result(supplied)
        token: Final = _NATIVE_MESSAGES_LOGGING.set(logging_future) if not logging_future.done() else None
        native: Final = execution.asynchronous(value, partial(_open_stream, logging_future=logging_future))
        try:
            result: Final = await _delivered(native(request))
            if logging_future.done() and not isinstance(result, AsyncIterator):
                logger: Final = logging_future.result()
                if logger.requires_success_await:
                    with anyio.CancelScope(shield=True):
                        await logger.await_success_accounting()
            return result
        finally:
            if token is not None:
                _NATIVE_MESSAGES_LOGGING.reset(token)

    return cast("NativeAmessages", amessages)  # cast-ok: the execution adapter returns the route's result


NATIVE_MESSAGES: Final = NativeBinding("messages", validate=_messages_binding)
NATIVE_AMESSAGES: Final = NativeBinding("amessages", validate=_amessages_binding)
