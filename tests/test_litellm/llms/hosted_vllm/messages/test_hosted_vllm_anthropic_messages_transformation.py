import pytest

import litellm
from litellm.llms.hosted_vllm.messages.transformation import HostedVLLMAnthropicMessagesConfig
from litellm.types.router import GenericLiteLLMParams

TEMPLATE_CONFIG = {
    "target": "chat_template_kwargs",
    "enabled": {"enable_thinking": True},
    "disabled": {"enable_thinking": False},
}
NATIVE_CONFIG = {
    "target": "native",
    "levels": {"low": ["minimal", "low"], "high": ["medium", "high"], "max": ["xhigh", "max"]},
    "disabled": "reject",
}
SWITCHABLE_NATIVE_CONFIG = {}


def transform(optional_params, reasoning_config=TEMPLATE_CONFIG, **litellm_params):
    return HostedVLLMAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 1024, **optional_params},
        litellm_params=GenericLiteLLMParams(model_info={"reasoning_effort": reasoning_config}, **litellm_params),
        headers={},
    )


@pytest.mark.parametrize(
    "optional_params, reasoning_config, extra_params, expected",
    [
        (
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}},
            TEMPLATE_CONFIG,
            {},
            {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "max"}},
        ),
        (
            {"thinking": {"type": "disabled"}},
            TEMPLATE_CONFIG,
            {},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ),
        (
            {"thinking": {"type": "disabled"}, "output_config": {"effort": "high"}},
            TEMPLATE_CONFIG,
            {},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ),
        (
            {"thinking": {"type": "adaptive"}},
            TEMPLATE_CONFIG,
            {},
            {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}},
        ),
        (
            {},
            TEMPLATE_CONFIG,
            {},
            {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}},
        ),
        (
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low", "format": {"type": "json"}}},
            NATIVE_CONFIG,
            {},
            {"output_config": {"effort": "low", "format": {"type": "json"}}},
        ),
        (
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
            TEMPLATE_CONFIG,
            {"chat_template_kwargs": {"enable_thinking": False}},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ),
    ],
)
def test_hosted_vllm_messages_reasoning_matrix(optional_params, reasoning_config, extra_params, expected):
    request = transform(optional_params, reasoning_config=reasoning_config, **extra_params)

    assert {key: request[key] for key in expected} == expected
    assert "thinking" not in request
    assert "reasoning_effort" not in request


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
            {},
            {"output_config": {"effort": "high"}},
        ),
    ],
)
def test_hosted_vllm_messages_native_switch_mapping(optional_params, expected):
    request = transform(optional_params, reasoning_config=SWITCHABLE_NATIVE_CONFIG)

    assert {key: request[key] for key in expected} == expected
    assert request.get("output_config", {}).get("effort") != "none"
