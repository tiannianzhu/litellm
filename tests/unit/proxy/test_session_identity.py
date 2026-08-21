from unittest.mock import AsyncMock, patch

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy.session_identity import apply_inferred_session_id


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
