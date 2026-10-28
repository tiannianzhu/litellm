"""
LIT-7701 attributes a pre_call_hook rejection's failure log to the model
group's single deployment (``model_id`` and ``custom_llm_provider``) and flags
it with ``PROXY_REJECTED_BEFORE_ROUTING_KEY``. The deployment health metrics
must keep treating such rejects as "no deployment picked": a key rate limit or
guardrail block never reached the deployment, so it must not flip
``litellm_deployment_state`` to partial outage or count as a deployment failure
response. A failure raised after the router picked a deployment (a post-call
guardrail block, a provider error) carries no flag and keeps its deployment labels.
"""

from datetime import datetime
from typing import Final
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY

import litellm
from litellm.constants import PROXY_REJECTED_BEFORE_ROUTING_KEY
from litellm.exceptions import GuardrailRaisedException, MalformedRequestFieldError
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.router_utils.cooldown_callbacks import router_cooldown_event_callback
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

    yield

    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def _attributed_failure_kwargs(exception: Exception, rejected_before_routing: bool) -> dict:
    return {
        "model": "openai/gpt-4.1",
        "litellm_params": {
            "custom_llm_provider": "openai",
            "metadata": {"model_info": {"id": "dep-1"}, "model_group": "internal-model"},
            **({PROXY_REJECTED_BEFORE_ROUTING_KEY: True} if rejected_before_routing else {}),
        },
        "standard_logging_object": {
            "model_id": "dep-1",
            "model_group": "internal-model",
            "api_base": "https://api.openai.com",
            "metadata": {},
        },
        "exception": exception,
    }


def _model_id_values(metric) -> set[str]:
    index = metric._labelnames.index("model_id")
    return {sample_key[index] for sample_key in metric._metrics}


class _ProviderError(Exception):
    status_code = 500


@pytest.mark.parametrize(
    "rejection",
    [
        HTTPException(status_code=403, detail="guardrail blocked"),
        ProxyException(message="budget exceeded", type="budget_exceeded", param=None, code=400),
        ProxyRateLimitError(detail={"error": "key rpm limit"}),
        GuardrailRaisedException(guardrail_name="pii", message="blocked", status_code=403),
    ],
    ids=["http_exception", "proxy_exception", "proxy_rate_limit", "guardrail_raised"],
)
def test_attributed_proxy_reject_leaves_deployment_healthy(rejection: Exception):
    logger = PrometheusLogger()

    logger.set_llm_deployment_failure_metrics(_attributed_failure_kwargs(rejection, rejected_before_routing=True))

    assert logger.litellm_deployment_state._metrics == {}
    assert _model_id_values(logger.litellm_deployment_failure_responses) == {""}
    assert _model_id_values(logger.litellm_deployment_total_requests) == {""}


@pytest.mark.parametrize(
    "failure",
    [
        _ProviderError("upstream 500"),
        GuardrailRaisedException(guardrail_name="pii", message="response blocked", status_code=400),
    ],
    ids=["provider_error", "post_call_guardrail"],
)
def test_failure_after_routing_still_marks_deployment_partial_outage(failure: Exception):
    logger = PrometheusLogger()

    logger.set_llm_deployment_failure_metrics(_attributed_failure_kwargs(failure, rejected_before_routing=False))

    assert _model_id_values(logger.litellm_deployment_state) == {"dep-1"}
    assert _model_id_values(logger.litellm_deployment_failure_responses) == {"dep-1"}


def test_malformed_field_rejection_after_routing_leaves_deployment_healthy() -> None:
    logger: Final = PrometheusLogger()
    rejection: Final = MalformedRequestFieldError(
        message="tools: Input should be a valid list", model="internal-model", llm_provider="openai"
    )

    logger.set_llm_deployment_failure_metrics(_attributed_failure_kwargs(rejection, rejected_before_routing=False))

    assert logger.litellm_deployment_state._metrics == {}
    assert _model_id_values(logger.litellm_deployment_failure_responses) == {""}


def _deployment_states() -> frozenset[tuple[str, str, str, str, float]]:
    metrics: Final = tuple(metric for metric in REGISTRY.collect() if metric.name == "litellm_deployment_state")
    assert len(metrics) == 1
    return frozenset(
        (
            sample.labels["model_id"],
            sample.labels["litellm_model_name"],
            sample.labels["api_base"],
            sample.labels["api_provider"],
            sample.value,
        )
        for sample in metrics[0].samples
    )


@pytest.mark.parametrize("endpoint", ["chat/completions", "embeddings"])
@pytest.mark.asyncio
async def test_cooldown_and_request_events_update_the_same_deployment_state(endpoint: str) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "public-model",
                "litellm_params": {"model": "openai/provider-model", "api_base": "https://gateway.example/v1"},
                "model_info": {"id": "dep-1"},
            },
            {
                "model_name": "public-model",
                "litellm_params": {"model": "openai/provider-model", "api_base": "https://gateway.example/v1"},
                "model_info": {"id": "dep-2"},
            },
        ]
    )
    logger: Final = PrometheusLogger()
    request_kwargs: Final = {
        "model": "provider-model",
        "litellm_params": {"metadata": {"model_info": {"id": "dep-1"}}},
        "standard_logging_object": {
            "model_id": "dep-1",
            "model_group": "public-model",
            "api_base": f"https://gateway.example/v1/{endpoint}",
            "hidden_params": {"additional_headers": {}},
            "metadata": {},
        },
    }
    state_1: Final = ("dep-1", "public-model", "https://gateway.example/v1", "openai")
    state_2: Final = ("dep-2", "public-model", "https://gateway.example/v1", "openai")

    with (
        patch("litellm.proxy.proxy_server.llm_router", router),  # test-quality-ok: inject the real proxy router
        patch("litellm._async_success_callback", []),
        patch("litellm.callbacks", [logger]),
    ):
        await router_cooldown_event_callback(router, "dep-1", 500, 60)
        await router_cooldown_event_callback(router, "dep-2", 500, 60)
        assert _deployment_states() == {(*state_1, 2.0), (*state_2, 2.0)}

        logger.set_llm_deployment_success_metrics(
            request_kwargs=request_kwargs,
            start_time=datetime(2026, 1, 1),
            end_time=datetime(2026, 1, 1, 0, 0, 1),
            enum_values=UserAPIKeyLabelValues(),
        )
        assert _deployment_states() == {(*state_1, 0.0), (*state_2, 2.0)}

        logger.set_llm_deployment_failure_metrics({**request_kwargs, "exception": _ProviderError("upstream 500")})
        assert _deployment_states() == {(*state_1, 1.0), (*state_2, 2.0)}


@pytest.mark.parametrize("has_router", [False, True], ids=["no-router", "unknown-deployment"])
def test_unresolved_deployment_state_keeps_callback_labels(has_router: bool) -> None:
    router: Final = litellm.Router(model_list=[]) if has_router else None
    logger: Final = PrometheusLogger()

    with patch("litellm.proxy.proxy_server.llm_router", router):  # test-quality-ok: inject the real proxy router
        logger.set_deployment_complete_outage("sdk-model", "sdk-id", "https://sdk.example/v1", "provider")
        logger.set_deployment_healthy("sdk-model", "sdk-id", "https://sdk.example/v1", "provider")

    assert _deployment_states() == {("sdk-id", "sdk-model", "https://sdk.example/v1", "provider", 0.0)}
