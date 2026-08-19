from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal, NoReturn, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

ReasoningTarget: TypeAlias = Literal["native", "chat_template_kwargs"]
ReasoningTemplateValue: TypeAlias = bool | str | int | float
ReasoningTemplateParams: TypeAlias = Mapping[str, ReasoningTemplateValue]
DisabledReasoningConfig: TypeAlias = Literal["reject"] | ReasoningTemplateParams

DEFAULT_ENABLED: Final[ReasoningTemplateParams] = MappingProxyType({"enable_thinking": True})
DEFAULT_DISABLED: Final[ReasoningTemplateParams] = MappingProxyType({"enable_thinking": False})
DEFAULT_LEVELS: Final = (
    ("low", ("minimal", "low")),
    ("high", ("medium", "high")),
    ("max", ("xhigh", "max")),
)


class ReasoningEffortValue(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    effort: str | None = None


class ReasoningEffortConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    target: ReasoningTarget = "native"
    enabled: ReasoningTemplateParams = Field(default_factory=lambda: DEFAULT_ENABLED)
    disabled: DisabledReasoningConfig = Field(default_factory=lambda: DEFAULT_DISABLED)
    levels: Mapping[str, Sequence[str]] | None = None
    default: str | None = "high"

    def normalize(
        self,
        value: object,
        model: str,
    ) -> str | None:
        requested_effort: Final = get_reasoning_effort_value(value)
        effort: Final = requested_effort if requested_effort is not None else self.default
        if effort is None:
            return None
        if effort == "none":
            if self.disabled == "reject":
                self._raise_unsupported(model, "This model always has reasoning enabled and cannot be disabled.")
            return "disabled"

        level_items: Final = tuple(self.levels.items()) if self.levels is not None else DEFAULT_LEVELS
        normalized: Final = next(
            (level for level, aliases in level_items if effort == level or effort in aliases),
            None,
        )
        if normalized is not None:
            return normalized
        if self.target == "native":
            self._raise_unsupported(model, f"Unsupported reasoning effort '{effort}'.")
        return None

    def get_chat_template_kwargs(self, effort: str, model: str) -> Mapping[str, ReasoningTemplateValue]:
        if effort != "disabled":
            return MappingProxyType({**self.enabled, "reasoning_effort": effort})
        disabled: Final = self.disabled
        if disabled == "reject":
            self._raise_unsupported(model, "This model always has reasoning enabled and cannot be disabled.")
        return MappingProxyType(dict(disabled))

    @staticmethod
    def _raise_unsupported(model: str, message: str) -> NoReturn:
        from litellm.exceptions import UnsupportedParamsError

        raise UnsupportedParamsError(
            model=model,
            llm_provider="hosted_vllm",
            message=message,
        )


def get_reasoning_effort_config(value: object) -> ReasoningEffortConfig | None:
    if value is None:
        return None
    if isinstance(value, ReasoningEffortConfig):
        return value
    return ReasoningEffortConfig.model_validate(value)


def get_reasoning_effort_value(value: object) -> str | None:
    if isinstance(value, Mapping):
        mapped_effort: Final = ReasoningEffortValue.model_validate(value).effort
        return mapped_effort.lower() if mapped_effort is not None else None
    return value.lower() if isinstance(value, str) else None
