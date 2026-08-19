import pytest

import litellm
from litellm.llms.hosted_vllm.messages.transformation import HostedVLLMAnthropicMessagesConfig
from litellm.types.router import GenericLiteLLMParams

NATIVE_CONFIG = {
    "levels": {"low": ["minimal", "low"], "high": ["medium", "high"], "max": ["xhigh", "max"]},
    "disabled": "reject",
}
SWITCHABLE_NATIVE_CONFIG = {}


def transform(optional_params, reasoning_config=SWITCHABLE_NATIVE_CONFIG, **litellm_params):
    return HostedVLLMAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 1024, **optional_params},
        litellm_params=GenericLiteLLMParams(model_info={"reasoning_effort": reasoning_config}, **litellm_params),
        headers={},
    )


def test_hosted_vllm_messages_rejects_disabling_native_reasoning():
    with pytest.raises(litellm.UnsupportedParamsError, match="always has reasoning enabled"):
        transform({"thinking": {"type": "disabled"}}, reasoning_config=NATIVE_CONFIG)


@pytest.mark.parametrize(
    "optional_params, expected",
    [
        (
            {"thinking": {"type": "disabled"}, "output_config": {"effort": "high"}},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ),
        (
            {"thinking": {"type": "adaptive"}},
            {
                "chat_template_kwargs": {"enable_thinking": True},
                "output_config": {"effort": "high"},
            },
        ),
        (
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}},
            {
                "chat_template_kwargs": {"enable_thinking": True},
                "output_config": {"effort": "max"},
            },
        ),
        (
            {"output_config": {"effort": "high"}},
            {"output_config": {"effort": "high"}},
        ),
        (
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            {"chat_template_kwargs": {"enable_thinking": True}, "output_config": {"effort": "low"}},
        ),
        (
            {"output_config": {"effort": "medium", "format": {"type": "json"}}},
            {"output_config": {"effort": "high", "format": {"type": "json"}}},
        ),
        (
            {},
            {"output_config": {"effort": "high"}},
        ),
    ],
)
def test_hosted_vllm_messages_native_switch_mapping(optional_params, expected):
    request = transform(optional_params, reasoning_config=SWITCHABLE_NATIVE_CONFIG)

    assert {key: request[key] for key in expected} == expected
    assert request.get("output_config", {}).get("effort") != "none"
    assert "thinking" not in request
    if optional_params.get("thinking", {}).get("type") == "disabled":
        assert "effort" not in request.get("output_config", {})
