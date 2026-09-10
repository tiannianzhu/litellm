from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

import pytest

from litellm.integrations.custom_logger_v2 import Field, Hook, InvalidDeclarationError, InvalidPatchError, Patch
from litellm.integrations.custom_logger_v2.derivations import cost, usage
from litellm.integrations.custom_logger_v2.field_sources import (
    FieldSource,
    HookArgs,
    SourcedFields,
    derived,
    mapped,
    source_table_problem,
    with_sources,
)
from litellm.integrations.custom_logger_v2.hook_point_fields import ResultFields
from litellm.integrations.custom_logger_v2.hooks import Subscription
from litellm.integrations.custom_logger_v2.legacy import KwargsFields
from litellm.types.integrations.custom_logger_v2 import Cost, Usage, UsageRecord

INNER: Final = MappingProxyType({Field.model: "gpt"})
ARGS: Final = HookArgs(request={"model": "gpt"})


def _counting(value: object) -> tuple[list[HookArgs], Callable[[HookArgs], object]]:
    calls: Final[list[HookArgs]] = []

    def read(args: HookArgs) -> object:
        calls.append(args)
        return value

    return calls, read


def test_a_derived_source_runs_only_when_read_and_then_once_for_every_reader_of_the_hook() -> None:
    calls, read = _counting({"key_hash": "h"})
    fields: Final = SourcedFields(INNER, {Field.key_identity: derived(read)}, ARGS)

    assert calls == []
    assert dict(fields) == {Field.model: "gpt"}
    assert calls == []
    first: Final = fields[Field.key_identity]
    second: Final = fields[Field.key_identity]

    assert first is second
    assert calls == [ARGS]
    assert dict(fields) == {Field.model: "gpt", Field.key_identity: first}


def test_the_derived_cache_survives_a_patch_to_another_field() -> None:
    calls, read = _counting({"key_hash": "h"})
    fields: Final = SourcedFields(INNER, {Field.key_identity: derived(read)}, ARGS)
    _ = fields[Field.key_identity]

    patched: Final = fields.apply(Patch().set(Field.model, "claude"))
    _ = patched[Field.key_identity]

    assert patched[Field.model] == "claude"
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("replacement_field", "source_field", "replacement", "expected"),
    [
        (
            Field.deprecated_kwargs,
            Field.cost,
            {"standard_logging_object": {"response_cost": 2.5}},
            Cost(response_cost=2.5),
        ),
        (
            Field.deprecated_response_obj,
            Field.usage,
            {"usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}},
            Usage(
                response=UsageRecord(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                logged=UsageRecord(prompt_tokens=3, completion_tokens=4, total_tokens=7),
            ),
        ),
    ],
)
def test_legacy_replacements_update_unread_sources_and_preserve_already_derived_values(
    replacement_field: Field,
    source_field: Field,
    replacement: Mapping[str, object],
    expected: Cost | Usage,
) -> None:
    request: Final = {
        "standard_logging_object": {
            "response_cost": 1.25,
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        }
    }
    response: Final = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    calls, read = _counting({"key_hash": "h"})
    carrier: Final = (
        ResultFields(request, {}, Field.response, response)
        if replacement_field is Field.deprecated_response_obj
        else KwargsFields(request, {Field.standard_logging_payload: "standard_logging_object"}, response)
    )
    fields: Final = SourcedFields(
        carrier,
        {Field.key_identity: derived(read), Field.cost: derived(cost), Field.usage: derived(usage)},
        HookArgs(request=request, result=response),
    )
    cached: Final = fields[Field.key_identity]

    patched: Final = fields.apply(Patch().set(replacement_field, replacement))

    assert patched[replacement_field] is replacement
    assert patched[Field.key_identity] is cached
    assert len(calls) == 1
    assert patched[source_field] == expected


def test_a_derived_source_that_finds_nothing_leaves_the_field_absent() -> None:
    calls, read = _counting(None)
    fields: Final = SourcedFields(INNER, {Field.key_identity: derived(read)}, ARGS)

    assert Field.key_identity not in fields
    assert Field.key_identity not in fields
    assert len(calls) == 1


def test_a_mapped_source_reads_its_carrier_every_time_and_writes_through_it() -> None:
    carrier: Final[dict[str, object]] = {"reserved_cost": 1.0}

    def write(args: HookArgs, value: object) -> None:
        assert args is ARGS
        assert isinstance(value, Mapping)
        carrier.update(value)

    fields: Final = SourcedFields(INNER, {Field.budget_reservation: mapped(lambda args: dict(carrier), write)}, ARGS)
    before: Final = fields[Field.budget_reservation]

    after: Final = fields.apply(Patch().set(Field.budget_reservation, {"reserved_cost": 2.0}))

    assert before == {"reserved_cost": 1.0}
    assert after is fields
    assert after[Field.budget_reservation] == {"reserved_cost": 2.0}


def test_a_patch_to_a_derived_source_is_rejected_and_its_value_stays() -> None:
    fields: Final = SourcedFields(INNER, {Field.key_identity: derived(lambda args: {"key_hash": "h"})}, ARGS)

    with pytest.raises(InvalidPatchError, match="derived from the hook's arguments"):
        fields.apply(Patch().set(Field.key_identity, {"key_hash": "forged"}))

    assert fields[Field.key_identity] == {"key_hash": "h"}


def test_a_source_table_cannot_make_a_derived_only_field_writable() -> None:
    table: Final[Mapping[Field, FieldSource]] = {Field.usage: mapped(lambda args: None, lambda args, value: None)}

    assert source_table_problem(table) == "usage are derived-only, so their sources cannot write"
    with pytest.raises(ValueError, match="derived-only"):
        SourcedFields(INNER, table, ARGS)


def test_no_sources_hands_back_the_inner_fields_unwrapped() -> None:
    assert with_sources(INNER, {}, ARGS) is INNER


@pytest.mark.parametrize(
    ("hook", "writes", "expected"),
    [
        (Hook.attempt_ended, frozenset({Field.usage}), "usage"),
        (Hook.pre_call, frozenset({Field.key_identity, Field.budget_plan}), "budget_plan, key_identity"),
        (Hook.log_prepare, frozenset({Field.standard_logging_payload_ref}), "standard_logging_payload_ref"),
    ],
)
def test_declaring_a_write_to_a_derived_only_field_is_a_manifest_error(
    hook: Hook, writes: frozenset[Field], expected: str
) -> None:
    with pytest.raises(InvalidDeclarationError) as raised:
        Subscription(hook=hook, handler="handler", writes=writes)

    assert str(raised.value) == (
        f"handler on {hook.value}: writes {expected}, which are derived from the hook's arguments and never writable"
    )


def test_the_pre_call_mapped_fields_are_writable_there() -> None:
    fields: Final = frozenset({Field.budget_reservation, Field.rate_limit_statuses})

    assert Subscription(hook=Hook.pre_call, handler="handler", reads=fields, writes=fields).writes == fields
