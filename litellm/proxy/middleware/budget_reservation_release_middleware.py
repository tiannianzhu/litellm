import asyncio
from collections.abc import Awaitable, Callable, Mapping
from functools import cache
from typing import Final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_SCOPES_AUTH_STAMPS: Final = frozenset({"http", "websocket"})


class BudgetReservationReleaseMiddleware:
    """Releases the budget reservation auth made for a request once no callback owns it.

    Auth stamps the reservation on the request or socket state; a call that starts
    claims it for the cost callbacks, which settle it on success or failure.
    Unclaimed reservations are released before the final HTTP body is sent or
    after a socket closes. Request exit also covers exceptions and cancellation.
    """

    def __init__(self, app: ASGIApp, release: Callable[[Mapping[str, object]], Awaitable[None]]) -> None:
        self.app = app
        self.release = release

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in _SCOPES_AUTH_STAMPS:
            await self.app(scope, receive, send)
            return

        async def release_reservation() -> None:
            state: Final = scope.get("state")
            budget_reservation: Final = state.get("budget_reservation") if isinstance(state, Mapping) else None
            if isinstance(budget_reservation, Mapping):
                await self.release(budget_reservation)

        @cache
        def release_task() -> asyncio.Task[None]:
            return asyncio.create_task(release_reservation())

        async def send_with_release(message: Message) -> None:
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                await asyncio.shield(release_task())
            await send(message)

        try:
            await self.app(scope, receive, send_with_release)
        finally:
            await asyncio.shield(release_task())
