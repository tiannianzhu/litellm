import asyncio
import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import SESSION_ID_OMITTED_METADATA_KEY
from litellm.proxy.litellm_pre_call_utils import apply_missing_session_id_policy
from litellm.proxy.session_identity import apply_inferred_session_id
from starlette.requests import Request


def _request(messages: list[dict], *, end_user: str = "user-a", model: str = "deepseek-v4-flash") -> dict:
    return {
        "model": model,
        "messages": messages,
        "metadata": {
            "user_api_key_hash": "shared-key",
            "user_api_key_end_user_id": end_user,
        },
    }


@pytest.mark.asyncio
async def test_prefix_fallback_groups_growing_direct_model_conversation():
    cache = DualCache()
    first_messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Analyze this repository."},
    ]
    continued_messages = [
        *first_messages,
        {"role": "assistant", "content": "I found the relevant module."},
        {"role": "user", "content": "Now fix it."},
    ]
    first = _request(first_messages)
    continued = _request(continued_messages)

    await apply_inferred_session_id(request_data=first, cache=cache)
    await apply_inferred_session_id(request_data=continued, cache=cache)

    assert first["metadata"]["session_id"].startswith("prefix-")
    assert continued["metadata"]["session_id"] == first["metadata"]["session_id"]


@pytest.mark.asyncio
async def test_prefix_fallback_groups_responses_api_conversation():
    from litellm.llms.openai.responses.guardrail_translation.handler import OpenAIResponsesHandler
    from litellm.types.utils import CallTypes

    cache = DualCache()
    first_input = [{"role": "user", "content": "Analyze this repository."}]
    continued_input = [
        *first_input,
        {"role": "assistant", "content": "I found the relevant module."},
        {"role": "user", "content": "Now fix it."},
    ]

    def request(input_messages: list[dict]) -> dict:
        return {
            "model": "deepseek-v4-flash",
            "input": input_messages,
            "litellm_metadata": {
                "user_api_key_hash": "shared-key",
                "user_api_key_end_user_id": "user-a",
                "user_api_key_request_route": "/v1/responses",
            },
        }

    first = request(first_input)
    continued = request(continued_input)
    with patch(  # test-quality-ok: response-route translation lookup has no injection seam
        "litellm.llms.load_guardrail_translation_mappings",
        return_value={CallTypes.responses: OpenAIResponsesHandler},
    ):
        await apply_inferred_session_id(request_data=first, cache=cache)
        await apply_inferred_session_id(request_data=continued, cache=cache)

    assert continued["litellm_metadata"]["session_id"] == first["litellm_metadata"]["session_id"]


@pytest.mark.asyncio
async def test_prefix_fallback_does_not_group_only_a_shared_system_prompt():
    cache = DualCache()
    system = {"role": "system", "content": "A shared coding-agent prompt."}
    first = _request([system, {"role": "user", "content": "Fix project A."}])
    second = _request([system, {"role": "user", "content": "Fix project B."}])

    await apply_inferred_session_id(request_data=first, cache=cache)
    await apply_inferred_session_id(request_data=second, cache=cache)

    assert first["metadata"]["session_id"] != second["metadata"]["session_id"]


@pytest.mark.asyncio
async def test_prefix_fallback_isolated_by_caller_and_model():
    cache = DualCache()
    messages = [{"role": "user", "content": "Review this change."}]
    first = _request(messages)
    other_user = _request(messages, end_user="user-b")
    other_model = _request(messages, model="deepseek-v4-pro")

    for request_data in (first, other_user, other_model):
        await apply_inferred_session_id(request_data=request_data, cache=cache)

    assert len({request["metadata"]["session_id"] for request in (first, other_user, other_model)}) == 3


@pytest.mark.asyncio
async def test_explicit_session_id_is_preserved_without_cache_access():
    cache = AsyncMock()
    request_data = _request([{"role": "user", "content": "Hello."}])
    request_data["metadata"]["session_id"] = "client-session"

    result = await apply_inferred_session_id(request_data=request_data, cache=cache)

    assert result is None
    assert request_data["metadata"]["session_id"] == "client-session"
    cache.async_batch_get_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_cache_failure_does_not_fail_the_request():
    cache = AsyncMock()
    cache.async_batch_get_cache.side_effect = RuntimeError("cache unavailable")
    request_data = _request([{"role": "user", "content": "Hello."}])

    result = await apply_inferred_session_id(request_data=request_data, cache=cache)

    assert result is None
    assert "session_id" not in request_data["metadata"]


@pytest.mark.asyncio
async def test_root_session_takes_precedence_over_inference() -> None:
    request_data: Final = _request([{"role": "user", "content": "Hello."}])
    request_data["litellm_session_id"] = "explicit-session"

    result: Final = await apply_inferred_session_id(request_data=request_data, cache=DualCache())

    assert result is None
    assert request_data["litellm_session_id"] == "explicit-session"
    assert "session_id" not in request_data["metadata"]


@pytest.mark.asyncio
async def test_omit_policy_prevents_prefix_inference() -> None:
    request_data: Final = _request([{"role": "user", "content": "Hello."}])
    apply_missing_session_id_policy(
        data=request_data,
        _metadata_variable_name="metadata",
        general_settings={"missing_session_id": "omit"},
        request=Request({"type": "http", "path": "/v1/chat/completions", "headers": []}),
    )

    result: Final = await apply_inferred_session_id(request_data=request_data, cache=DualCache())

    assert result is None
    assert request_data["metadata"][SESSION_ID_OMITTED_METADATA_KEY] is True
    assert "session_id" not in request_data["metadata"]


@pytest.mark.asyncio
async def test_anthropic_system_prompt_is_part_of_conversation_fingerprint() -> None:
    cache: Final = DualCache()
    first: Final = _request([{"role": "user", "content": "Hello."}])
    first["system"] = "Answer about project A."
    second: Final = _request(first["messages"])
    second["system"] = "Answer about project B."

    first_session: Final = await apply_inferred_session_id(request_data=first, cache=cache)
    second_session: Final = await apply_inferred_session_id(request_data=second, cache=cache)

    assert first_session is not None
    assert second_session is not None
    assert first_session != second_session


@pytest.mark.asyncio
async def test_concurrent_prefix_inference_uses_shared_cache_winner() -> None:
    redis_values: Final[dict[str, str]] = {}  # mutable-ok: emulates the Redis server at its client boundary
    arrivals: Final[asyncio.Queue[None]] = asyncio.Queue()
    both_read: Final = asyncio.Event()

    async def mget(*, keys: list[str]) -> list[str | None]:
        snapshot: Final = [redis_values.get(key) for key in keys]
        arrivals.put_nowait(None)
        if arrivals.qsize() == 2:
            both_read.set()
        await both_read.wait()
        return snapshot

    async def set_value(*, name: str, value: str, nx: bool, ex: int) -> bool | None:
        if nx and name in redis_values:
            return None
        redis_values[name] = value
        return True

    async def get_value(key: str) -> str | None:
        return redis_values.get(key)

    redis_client: Final = AsyncMock()
    redis_client.mget.side_effect = mget
    redis_client.set.side_effect = set_value
    redis_client.get.side_effect = get_value
    with (
        patch("litellm._redis.get_redis_client", return_value=MagicMock()),
        patch("litellm._redis.get_redis_async_client", return_value=redis_client),
        patch("litellm.in_memory_llm_clients_cache", new=InMemoryCache()),
    ):
        first_cache: Final = DualCache(redis_cache=RedisCache(host="session-regression.invalid"))
        second_cache: Final = DualCache(redis_cache=RedisCache(host="session-regression.invalid"))
        first: Final = _request([{"role": "user", "content": "Hello."}])
        second: Final = _request(first["messages"])

        sessions: Final = await asyncio.gather(
            apply_inferred_session_id(request_data=first, cache=first_cache),
            apply_inferred_session_id(request_data=second, cache=second_cache),
        )

    assert sessions[0] is not None
    assert sessions[0] == sessions[1]
    assert {json.loads(value) for value in redis_values.values()} == {sessions[0]}
