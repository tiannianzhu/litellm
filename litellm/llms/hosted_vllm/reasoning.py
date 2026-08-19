from collections.abc import Mapping, Sequence
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.reasoning_effort_capability import (
    DeploymentReasoningEffort,
    RequestedReasoningEffort,
    deployment_reasoning_effort_levels,
    resolve_deployment_reasoning_effort,
)
from litellm.litellm_core_utils.reasoning_effort_strength import (
    REASONING_EFFORT_STRENGTH_ORDER,
    nearest_declared_reasoning_effort,
)
from litellm.litellm_core_utils.reasoning_effort_utils import reasoning_effort_from_thinking

_DEFAULT_LEVELS: Final = ("low", "high", "max")


class ReasoningEffortConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    disabled: Literal["reject"] | None = None
    levels: Mapping[str, Sequence[str]] | None = None


def hosted_vllm_reasoning_policy(custom_llm_provider: str | None, model_info: object) -> object:
    info: Final = as_str_mapping(model_info)
    return info.get("reasoning_effort") if custom_llm_provider == "hosted_vllm" and info is not None else None


def reasoning_policy_drop_paths(paths: object, *protected: str) -> Sequence[str] | None:
    if not isinstance(paths, (list, tuple)):
        return None
    names: Final = (path for path in paths if isinstance(path, str))
    return [path for path in names if path not in protected]


def resolve_hosted_vllm_reasoning_effort(
    requested: RequestedReasoningEffort,
    *,
    model: str,
    custom_llm_provider: str | None,
    deployment_model_info: object,
    model_group: str | None,
    drop_params: bool,
    thinking: object = None,
    extra_body: object = None,
) -> DeploymentReasoningEffort | None:
    policy: Final = hosted_vllm_reasoning_policy(custom_llm_provider, deployment_model_info)
    if policy is None:
        return None
    extra: Final = as_str_mapping(extra_body)
    if extra is not None and any(
        key in extra for key in ("reasoning_effort", "reasoning", "thinking", "output_config")
    ):
        raise UnsupportedParamsError(
            model=model,
            llm_provider="hosted_vllm",
            message="Send reasoning controls as request fields instead of in extra_body.",
        )
    config: Final = ReasoningEffortConfig.model_validate(policy)
    fields: Final = as_str_mapping(requested)
    raw_effort: Final = fields.get("effort") if fields is not None else requested
    thinking_fields: Final = as_str_mapping(thinking)
    derived_effort: Final = (
        "none"
        if raw_effort is None and thinking_fields is not None and thinking_fields.get("type") == "disabled"
        else reasoning_effort_from_thinking(thinking, model=model, drop_params=drop_params)
        if raw_effort is None
        else None
    )
    effort: Final = raw_effort.lower() if isinstance(raw_effort, str) else (derived_effort or "high")
    if effort == "none" and config.disabled == "reject":
        raise UnsupportedParamsError(
            model=model,
            llm_provider="hosted_vllm",
            message="This model always has reasoning enabled and cannot be disabled.",
        )

    declared: Final = deployment_reasoning_effort_levels(deployment_model_info)
    levels: Final = (
        declared if declared is not None else (tuple(config.levels) if config.levels is not None else _DEFAULT_LEVELS)
    )
    configured_target: Final = (
        next(
            (level for level, aliases in config.levels.items() if effort == level or effort in aliases),
            None,
        )
        if config.levels is not None
        else None
    )
    allowed: Final = (
        effort in REASONING_EFFORT_STRENGTH_ORDER or effort in levels
        if config.levels is None or declared is not None
        else configured_target is not None
    )
    if (raw_effort is not None and not isinstance(raw_effort, str)) or (effort != "none" and not allowed):
        raise UnsupportedParamsError(
            model=model, llm_provider="hosted_vllm", message=f"Unsupported reasoning effort '{effort}'."
        )

    legacy_alias: Final = (
        configured_target
        if declared is None
        and configured_target is not None
        and configured_target != nearest_declared_reasoning_effort(effort, levels)
        else None
    )
    request_level: Final = legacy_alias or effort
    effective_info: Final = {
        **(as_str_mapping(deployment_model_info) or {}),
        "reasoning_effort_levels": levels,
        **({"reasoning_effort_rounding": "up"} if declared is None else {}),
    }
    return resolve_deployment_reasoning_effort(
        {**fields, "effort": request_level} if fields is not None else request_level,
        model=model,
        custom_llm_provider=custom_llm_provider,
        deployment_model_info=effective_info,
        model_group=model_group,
        drop_params=False,
    )
