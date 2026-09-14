from __future__ import annotations

import asyncio
import gc
import weakref
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final, Literal

import httpx
import pytest
import pytest_asyncio
from aiohttp import ClientSession, TraceConfig, TraceRequestStartParams, web
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.caching.evicted_client_closer import EvictedClientCloser
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig
from litellm.types.utils import ModelResponse


@pytest_asyncio.fixture(loop_scope="function")
async def gated_stream_server() -> AsyncIterator[tuple[str, asyncio.Event]]:
    release: Final = asyncio.Event()

    async def stream(request: web.Request) -> web.StreamResponse:
        response: Final = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: first\n\n")
        await release.wait()
        if request.query.get("abort"):
            transport: Final = request.transport
            assert transport is not None
            transport.close()
            return response
        try:
            await response.write(b"data: last\n\n")
            await response.write_eof()
        except ConnectionResetError:
            pass
        return response

    app: Final = web.Application()
    app.router.add_get("/", stream)
    runner: Final = web.AppRunner(app)
    await runner.setup()
    try:
        site: Final = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        _, port = runner.addresses[0]
        yield f"http://127.0.0.1:{port}/", release
    finally:
        release.set()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["complete", "close", "cancel", "error"])
async def test_evicted_handler_keeps_concurrent_streams_alive(
    gated_stream_server: tuple[str, asyncio.Event], ending: Literal["complete", "close", "cancel", "error"]
) -> None:
    url, release = gated_stream_server
    cache: Final = LLMClientCache(evicted_client_closer=EvictedClientCloser(grace_seconds=0))
    handler: Final = AsyncHTTPHandler()
    cache.set_cache("stream-owner", handler, litellm_owned_client=True)
    first: Final = await handler.client.send(handler.client.build_request("GET", url), stream=True)
    second_url: Final = url + "?abort=1" if ending == "error" else url
    second: Final = await handler.client.send(handler.client.build_request("GET", second_url), stream=True)
    iterator: Final = second.aiter_bytes()
    assert await anext(iterator) == b"data: first\n\n"
    session: Final = handler.client._transport.client
    owner: Final = weakref.ref(handler)
    cache._remove_key(cache.update_cache_key_with_event_loop("stream-owner"))
    del handler
    gc.collect()
    await asyncio.sleep(0)
    try:
        assert owner() is not None
        assert not session.closed
        await first.aclose()
        cache.evicted_client_closer.reap()
        await asyncio.sleep(0)
        assert owner() is not None
        assert not session.closed
        if ending == "complete":
            release.set()
            assert await asyncio.wait_for(anext(iterator), 2) == b"data: last\n\n"
            with pytest.raises(StopAsyncIteration):
                await anext(iterator)
        elif ending == "cancel":
            pending: Final = asyncio.create_task(anext(iterator))
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        elif ending == "error":
            release.set()
            with pytest.raises(httpx.ReadError):
                await asyncio.wait_for(anext(iterator), 2)
        else:
            await second.aclose()
        gc.collect()
        await asyncio.sleep(0)
        assert owner() is None
        assert session.closed
    finally:
        await first.aclose()
        await second.aclose()
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_fails", [False, True])
async def test_stream_owner_preserves_hooks_and_releases_nonstream_responses(
    gated_stream_server: tuple[str, asyncio.Event], hook_fails: bool
) -> None:
    url, release = gated_stream_server
    observed: Final[list[str | int]] = []

    async def on_request(request: httpx.Request) -> None:
        observed.append(request.method)

    async def on_response(response: httpx.Response) -> None:
        observed.append(response.status_code)
        if hook_fails:
            raise ValueError("fixture hook error")
        await response.aread()

    hooks: Final = {"request": [on_request], "response": [on_response]}
    handler: Final = AsyncHTTPHandler(event_hooks=hooks)
    owner: Final = weakref.ref(handler)
    release.set()
    try:
        if hook_fails:
            with pytest.raises(ValueError, match="fixture hook error"):
                await handler.client.get(url)
        else:
            response: Final = await handler.client.get(url)
            assert response.content == b"data: first\n\ndata: last\n\n"
            assert response.is_closed
        assert observed == ["GET", 200]
        assert hooks == {"request": [on_request], "response": [on_response]}
    finally:
        session: Final = handler.client._transport.client
        del handler
        gc.collect()
        await asyncio.sleep(0)
        try:
            assert owner() is None
            assert session.closed
        finally:
            await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fake_stream", (False, True))
async def test_async_streaming_reuses_the_provided_aiohttp_session(
    monkeypatch: pytest.MonkeyPatch, fake_stream: bool
) -> None:
    async def completion_stream(request: web.Request) -> web.StreamResponse:
        body: Final = TypeAdapter(dict[str, JsonValue]).validate_python(await request.json())
        assert body.get("stream", False) is (not fake_stream)
        if fake_stream:
            return web.json_response(
                {
                    "id": "chatcmpl-session",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )
        response: Final = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
            b'"model":"test-model","choices":[{"index":0,"delta":{"content":"hi"},'
            b'"finish_reason":null}]}\n\n'
            b"data: [DONE]\n\n"
        )
        await response.write_eof()
        return response

    app: Final = web.Application()
    app.router.add_post("/chat/completions", completion_stream)
    runner: Final = web.AppRunner(app)
    await runner.setup()
    shared_request_urls: Final[list[str]] = []
    trace_config: Final = TraceConfig()

    async def record_shared_session_request(
        session: ClientSession, context: SimpleNamespace, params: TraceRequestStartParams
    ) -> None:
        shared_request_urls.append(str(params.url))

    trace_config.on_request_start.append(record_shared_session_request)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    try:
        site: Final = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        _, port = runner.addresses[0]
        url: Final = f"http://127.0.0.1:{port}/chat/completions"
        async with ClientSession(trace_configs=[trace_config]) as shared_session:
            logging_obj: Final = Logging(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
                call_type="acompletion",
                start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                litellm_call_id="shared-session-stream",
                function_id="f",
            )
            logging_obj.update_environment_variables(
                model="test-model", user="", optional_params={}, litellm_params={}, custom_llm_provider="hosted_vllm"
            )
            stream: Final = await BaseLLMHTTPHandler().completion(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                api_base=url,
                custom_llm_provider="hosted_vllm",
                model_response=ModelResponse(),
                encoding=None,
                logging_obj=logging_obj,
                optional_params={},
                timeout=5,
                litellm_params={},
                acompletion=True,
                stream=True,
                provider_config=HostedVLLMChatConfig(),
                shared_session=shared_session,
                fake_stream=fake_stream,
            )
            chunks: Final = [chunk async for chunk in stream]

            assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "hi"
            assert shared_request_urls == [url]
            assert not shared_session.closed
    finally:
        await runner.cleanup()
