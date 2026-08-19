"""
Utilities for handling OpenAI Responses API 'custom' tools (freeform/grammar tools)
when bridging to Chat Completions providers.

Custom tools are defined with ``type: "custom"`` and a grammar/format specification.
Since most Chat Completions providers only support standard ``function`` tools,
the bridge converts them to ``function`` tools with a single ``content`` string
parameter. When the model responds with a ``function_call`` for such a tool, this
module converts it back to the ``custom_tool_call`` format expected by clients like
Codex CLI.

The forward direction (custom -> function) and reverse direction (function_call ->
custom_tool_call) are both handled here so future custom tool types can be added by
extending this module without touching the streaming iterator or transformation
logic.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from litellm.types.llms.openai import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_MAX_ARGUMENTS_LEN: Final = 1_000_000
_CUSTOM_TOOL_WIRE_NAME_PREFIX: Final = "litellm_custom__"

TOOL_CALL_ITEM_ID_PREFIX_BY_TYPE: Final = MappingProxyType({"function_call": "fc", "custom_tool_call": "ctc"})
NativeResponsesCustomToolNameMap: TypeAlias = Mapping[str, tuple[str, str | None]]
_JSON_OBJECT_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_JSON_OBJECT_SEQUENCE_ADAPTER: Final[TypeAdapter[Sequence[Mapping[str, object]]]] = TypeAdapter(
    Sequence[Mapping[str, object]]
)
_JSON_VALUE_ADAPTER: Final[TypeAdapter[object]] = TypeAdapter(object)
_JSON_REQUEST_ADAPTER: Final = TypeAdapter(dict[str, JsonValue])


def _readonly_mapping(entries: tuple[tuple[str, object], ...]) -> Mapping[str, object]:
    return MappingProxyType(dict(entries))


def _json_request_payload(request: Mapping[str, object]) -> Mapping[str, object]:
    encoded: Final = json.dumps(request, default=_JSON_OBJECT_ADAPTER.validate_python)
    return _JSON_REQUEST_ADAPTER.validate_json(encoded)


def _json_object(value: object) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        return None


def _json_object_sequence(value: object) -> tuple[Mapping[str, object], ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        return tuple(_JSON_OBJECT_SEQUENCE_ADAPTER.validate_python(value, strict=True))
    except ValidationError:
        return None


def openai_shaped_tool_call_item_id(item_type: str, tool_id: str) -> str:
    prefix: Final = TOOL_CALL_ITEM_ID_PREFIX_BY_TYPE.get(item_type)
    if prefix is None or not tool_id or tool_id.startswith(prefix):
        return tool_id
    return f"{prefix}_{tool_id}"


class _ToolNameFields(BaseModel):
    type: str = ""
    name: str = ""
    tools: tuple[object, ...] = ()


def _tool_name_fields_of(tool: object) -> _ToolNameFields | None:
    try:
        return _ToolNameFields.model_validate(tool)
    except ValidationError:
        return None


def _custom_tool_name_of(tool: object) -> str | None:
    parsed: Final = _tool_name_fields_of(tool)
    if parsed is None or parsed.type != "custom" or not parsed.name:
        return None
    return parsed.name


def _nested_tools_of(tool: object) -> tuple[object, ...]:
    parsed: Final = _tool_name_fields_of(tool)
    if parsed is None or parsed.type != "namespace":
        return ()
    return parsed.tools


def extract_custom_tool_names(tools: Sequence[object] | None) -> set[str]:
    """Extract names of ``type: "custom"`` tools, at the top level or one level inside a ``namespace`` tool."""
    top_level: Final = tuple(tools or ())
    nested: Final = tuple(nested_tool for tool in top_level for nested_tool in _nested_tools_of(tool))
    return {name for tool in (*top_level, *nested) if (name := _custom_tool_name_of(tool)) is not None}


def is_custom_tool_call(tool_name: str, custom_tool_names: set[str]) -> bool:
    """Check if a tool call name corresponds to a custom tool."""
    return tool_name in custom_tool_names


def serialize_tool_call_arguments(raw_arguments: object, default: str = "") -> str:
    """Render tool call arguments as the JSON string tool-call schemas require.

    Arguments normally arrive already JSON-encoded, but clients and providers
    also send the decoded object. ``str()`` on a dict yields a Python repr with
    single quotes, which every downstream JSON parser rejects with errors like
    "Expecting ',' delimiter".
    """
    if isinstance(raw_arguments, str):
        return raw_arguments or default
    if raw_arguments is None:
        return default
    return json.dumps(raw_arguments, default=str)


def unwrap_custom_tool_arguments(arguments: str) -> str:
    """Extract the raw content string from JSON-wrapped arguments.

    The bridge converts custom tools to function tools with schema
    ``{"properties": {"content": {"type": "string"}}}``, so the model returns
    arguments like ``{"content": "*** Begin Patch\\n..."}``. This function
    extracts just the content string. If the arguments are not valid JSON or do
    not contain a ``content`` key, the original string is returned unchanged.
    """
    if not arguments:
        return ""
    if len(arguments) > _MAX_ARGUMENTS_LEN:
        return arguments
    try:
        parsed: Final = _JSON_VALUE_ADAPTER.validate_json(arguments)
        parsed_object: Final = _json_object(parsed)
        if parsed_object is not None and "content" in parsed_object:
            return str(parsed_object["content"])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return arguments


def unwrap_custom_tool_arguments_strict(arguments: object) -> str:
    """Return validated content from a function-wrapped custom call."""
    if not isinstance(arguments, str):
        raise TypeError("custom tool arguments must be a JSON string")
    if len(arguments) > _MAX_ARGUMENTS_LEN:
        raise ValueError("custom tool arguments exceed the maximum supported size")
    try:
        parsed: Final = _JSON_VALUE_ADAPTER.validate_json(arguments)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError("custom tool arguments must contain valid JSON") from exc
    parsed_object: Final = _json_object(parsed)
    if parsed_object is None:
        raise ValueError("custom tool arguments must be a JSON object with a string content field")
    if "content" not in parsed_object:
        raise ValueError("custom tool arguments must include a content field")
    content: Final[object] = parsed_object["content"]
    if not isinstance(content, str):
        raise TypeError("custom tool arguments content must be a string")
    return content


def _tools_from_request(request: Mapping[str, object]) -> tuple[Mapping[str, object], ...] | None:
    return _json_object_sequence(request.get("tools"))


def _custom_tool_identities(tools: Sequence[Mapping[str, object]]) -> tuple[tuple[str, str | None], ...]:
    top_level: Final = tuple(
        (name, None)
        for tool in tools
        if tool.get("type") == "custom"
        for name in (tool.get("name"),)
        if isinstance(name, str)
    )
    namespaced: Final = tuple(
        (name, namespace)
        for tool in tools
        if tool.get("type") == "namespace"
        for namespace in (tool.get("name"),)
        if isinstance(namespace, str)
        for nested_tools in (tool.get("tools"),)
        for nested_tool in (_json_object_sequence(nested_tools),)
        if nested_tool is not None
        for nested_tool_item in nested_tool
        if nested_tool_item.get("type") == "custom"
        for name in (nested_tool_item.get("name"),)
        if isinstance(name, str)
    )
    return top_level + namespaced


def _ordinary_function_wire_names(tools: Sequence[Mapping[str, object]]) -> frozenset[str]:
    top_level_names: Final = frozenset(
        name
        for tool in tools
        if tool.get("type") == "function"
        for name in (tool.get("name"),)
        if isinstance(name, str)
    )
    namespaced_names: Final = frozenset(
        f"{namespace}__{name}"
        for tool in tools
        if tool.get("type") == "namespace"
        for namespace in (tool.get("name"),)
        if isinstance(namespace, str)
        for nested_tools in (_json_object_sequence(tool.get("tools")),)
        if nested_tools is not None
        for nested_tool in nested_tools
        if nested_tool.get("type") == "function"
        for name in (nested_tool.get("name"),)
        if isinstance(name, str)
    )
    return top_level_names | namespaced_names


def _native_responses_custom_tool_name_map(
    tools: Sequence[Mapping[str, object]],
) -> NativeResponsesCustomToolNameMap:
    identities: Final = _custom_tool_identities(tools)
    if len(identities) != len(frozenset(identities)):
        raise ValueError("custom tool names must be unique within each namespace")
    ordinary_names: Final = _ordinary_function_wire_names(tools)
    candidates: Final = tuple(f"{namespace}__{name}" if namespace else name for name, namespace in identities)
    entries: Final = tuple(
        (
            candidate
            if candidate not in ordinary_names and candidates.count(candidate) == 1
            else _collision_safe_custom_tool_wire_name(name, namespace),
            (name, namespace),
        )
        for (name, namespace), candidate in zip(identities, candidates, strict=True)
    )
    wire_names: Final = frozenset(wire_name for wire_name, _ in entries)
    if len(wire_names) != len(entries) or wire_names & ordinary_names:
        raise ValueError("custom tool wire names conflict with function tools")
    return MappingProxyType(dict(entries))


def _collision_safe_custom_tool_wire_name(name: str, namespace: str | None) -> str:
    identity: Final = f"{namespace or ''}\x00{name}"
    digest: Final = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return f"{_CUSTOM_TOOL_WIRE_NAME_PREFIX}{digest}"


def native_responses_custom_tool_name_map(request: Mapping[str, object]) -> NativeResponsesCustomToolNameMap:
    """Map native Responses wire names to their original custom tool identities."""
    return _native_responses_custom_tool_name_map(_tools_from_request(request) or ())


def native_responses_namespace_tool_name_map(request: Mapping[str, object]) -> Mapping[str, tuple[str, str]]:
    """Return qualified namespace function names for a native Responses request."""
    from .transformation import LiteLLMCompletionResponsesConfig

    names: Final = LiteLLMCompletionResponsesConfig.namespace_tool_name_map(_tools_from_request(request))
    return MappingProxyType(
        {wire_name: identity for wire_name, identity in names.items() if wire_name == f"{identity[0]}__{identity[1]}"}
    )


def _wire_name_for_custom_tool(
    name: object,
    namespace: object,
    custom_tool_names: NativeResponsesCustomToolNameMap,
) -> str | None:
    if not isinstance(name, str) or (namespace is not None and not isinstance(namespace, str)):
        return None
    identity: Final = (name, namespace)
    mapped_wire_name: Final = next(
        (wire_name for wire_name, candidate in custom_tool_names.items() if candidate == identity), None
    )
    if mapped_wire_name is not None:
        return mapped_wire_name
    return _wire_name_without_definition(name, namespace)


def _flat_custom_function_tool(tool: Mapping[str, object], wire_name: str) -> Mapping[str, object]:
    converted: Final = convert_custom_tool_to_function_tool(MappingProxyType({**tool, "name": wire_name}))
    if converted is None:
        raise ValueError("expected a custom tool")
    function: Final = _json_object(converted.get("function"))
    if function is None:
        raise ValueError("custom tool conversion did not produce a function")
    allowed_callers: Final[object] = converted.get("allowed_callers")
    return _readonly_mapping(
        (("type", "function"), *function.items())
        + (("allowed_callers", allowed_callers),) * (allowed_callers is not None)
    )


def _normalize_native_responses_tools(
    tools: Sequence[Mapping[str, object]],
    custom_tool_names: NativeResponsesCustomToolNameMap,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        normalized_tool
        for tool in tools
        for normalized_tool in _normalize_native_responses_tool(tool, custom_tool_names)
    )


def _normalize_native_responses_tool(
    tool: Mapping[str, object],
    custom_tool_names: NativeResponsesCustomToolNameMap,
) -> tuple[Mapping[str, object], ...]:
    tool_type: Final[object] = tool.get("type")
    if tool_type == "custom":
        wire_name: Final = _wire_name_for_custom_tool(tool.get("name"), None, custom_tool_names)
        if wire_name is None:
            raise ValueError("custom tools must include a string name")
        return (_flat_custom_function_tool(tool, wire_name),)
    if tool_type != "namespace":
        return (_readonly_mapping(tuple(tool.items())),)
    namespace: Final[object] = tool.get("name")
    nested_tools: Final = _json_object_sequence(tool.get("tools"))
    if not isinstance(namespace, str) or nested_tools is None:
        return (_readonly_mapping(tuple(tool.items())),)
    custom_nested_tools: Final = tuple(
        nested_tool for nested_tool in nested_tools if nested_tool.get("type") == "custom"
    )
    converted_custom_tools: Final = tuple(
        _flat_custom_function_tool(nested_tool, wire_name)
        for nested_tool in custom_nested_tools
        for wire_name in (_wire_name_for_custom_tool(nested_tool.get("name"), namespace, custom_tool_names),)
        if wire_name is not None
    )
    if len(converted_custom_tools) != len(custom_nested_tools):
        raise ValueError("custom tools must include a string name")
    ordinary_nested_tools: Final = tuple(
        nested_tool for nested_tool in nested_tools if nested_tool.get("type") == "function"
    )
    flattened_ordinary_tools: Final = _flatten_namespace_function_tools(tool, ordinary_nested_tools)
    retained_nested_tools: Final = tuple(
        MappingProxyType(dict(nested_tool))
        for nested_tool in nested_tools
        if nested_tool.get("type") not in ("custom", "function")
    )
    retained_namespace: Final = (
        (MappingProxyType({**tool, "tools": retained_nested_tools}),) if retained_nested_tools else ()
    )
    return converted_custom_tools + flattened_ordinary_tools + retained_namespace


def _flatten_namespace_function_tools(
    namespace_tool: Mapping[str, object],
    nested_tools: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    if not nested_tools:
        return ()
    from .transformation import LiteLLMCompletionResponsesConfig

    forms: Final = LiteLLMCompletionResponsesConfig.responses_tools_to_chat_forms(
        (MappingProxyType({**namespace_tool, "description": "", "tools": nested_tools}),)
    )
    chat_tools: Final = tuple(chat_tool for form in forms for chat_tool in form.chat_tools)
    return tuple(
        MappingProxyType(tool)
        for tool in LiteLLMCompletionResponsesConfig.transform_chat_completion_tool_params_to_responses_api_tools(
            chat_tools
        )
    )


def _normalize_native_responses_input_item(
    item: Mapping[str, object],
    custom_tool_names: NativeResponsesCustomToolNameMap,
    namespace_tool_names: Mapping[str, tuple[str, str]],
) -> Mapping[str, object]:
    item_type: Final[object] = item.get("type")
    if item_type == "custom_tool_call":
        custom_wire_name: Final = _wire_name_for_custom_tool(item.get("name"), item.get("namespace"), custom_tool_names)
        raw_input: Final[object] = item.get("input")
        if not isinstance(raw_input, str):
            raise ValueError("custom tool call input must be a string")
        return MappingProxyType(
            {
                **MappingProxyType(
                    {
                        key: value
                        for key, value in item.items()
                        if key != "input" and (key != "namespace" or custom_wire_name is None)
                    }
                ),
                "type": "function_call",
                **(
                    MappingProxyType({"name": custom_wire_name})
                    if custom_wire_name is not None
                    else MappingProxyType({})
                ),
                "arguments": json.dumps(
                    MappingProxyType({"content": raw_input}), default=_JSON_OBJECT_ADAPTER.validate_python
                ),
            }
        )
    if item_type == "custom_tool_call_output":
        return MappingProxyType({**item, "type": "function_call_output"})
    if item_type == "function_call":
        function_wire_name: Final = _wire_name_for_namespace_function(
            item.get("name"), item.get("namespace"), namespace_tool_names
        )
        return MappingProxyType(
            {
                **MappingProxyType(
                    {key: value for key, value in item.items() if key != "namespace" or function_wire_name is None}
                ),
                **(
                    MappingProxyType({"name": function_wire_name})
                    if function_wire_name is not None
                    else MappingProxyType({})
                ),
            }
        )
    return MappingProxyType(dict(item))


def _normalize_native_responses_tool_choice(
    tool_choice: object,
    custom_tool_names: NativeResponsesCustomToolNameMap,
    namespace_tool_names: Mapping[str, tuple[str, str]],
) -> object:
    normalized_tool_choice: Final = _json_object(tool_choice)
    if normalized_tool_choice is None:
        return tool_choice
    tool_choice_type: Final[object] = normalized_tool_choice.get("type")
    if tool_choice_type == "custom":
        custom_wire_name: Final = _wire_name_for_custom_tool(
            normalized_tool_choice.get("name"), normalized_tool_choice.get("namespace"), custom_tool_names
        )
        return MappingProxyType(
            {
                **MappingProxyType(
                    {
                        key: value
                        for key, value in normalized_tool_choice.items()
                        if key != "namespace" or custom_wire_name is None
                    }
                ),
                "type": "function",
                **(
                    MappingProxyType({"name": custom_wire_name})
                    if custom_wire_name is not None
                    else MappingProxyType({})
                ),
            }
        )
    if tool_choice_type == "function":
        function_wire_name: Final = _wire_name_for_namespace_function(
            normalized_tool_choice.get("name"), normalized_tool_choice.get("namespace"), namespace_tool_names
        )
        return MappingProxyType(
            {
                **MappingProxyType(
                    {
                        key: value
                        for key, value in normalized_tool_choice.items()
                        if key != "namespace" or function_wire_name is None
                    }
                ),
                **(
                    MappingProxyType({"name": function_wire_name})
                    if function_wire_name is not None
                    else MappingProxyType({})
                ),
            }
        )
    if tool_choice_type != "allowed_tools":
        return normalized_tool_choice
    allowed_tools: Final = _json_object_sequence(normalized_tool_choice.get("tools"))
    if allowed_tools is None:
        return normalized_tool_choice
    normalized_allowed_tools: Final = tuple(
        _normalize_native_responses_tool_choice(allowed_tool, custom_tool_names, namespace_tool_names)
        for allowed_tool in allowed_tools
    )
    return _readonly_mapping((*normalized_tool_choice.items(), ("tools", normalized_allowed_tools)))


def _native_namespace_description(
    tool: Mapping[str, object], custom_tool_names: NativeResponsesCustomToolNameMap
) -> str | None:
    namespace: Final = tool.get("name")
    description: Final = tool.get("description")
    nested_tools: Final = _json_object_sequence(tool.get("tools"))
    if (
        tool.get("type") != "namespace"
        or not isinstance(namespace, str)
        or not isinstance(description, str)
        or not description
        or nested_tools is None
    ):
        return None
    wire_names: Final = tuple(
        wire_name for wire_name, identity in custom_tool_names.items() if identity[1] == namespace
    ) + tuple(
        f"{namespace}__{name}"
        for nested_tool in nested_tools
        if nested_tool.get("type") == "function"
        for name in (nested_tool.get("name"),)
        if isinstance(name, str)
    )
    if not wire_names:
        return None
    return f"Tool namespace {json.dumps(namespace, ensure_ascii=False)} ({', '.join(wire_names)}):\n{description}"


def normalize_native_responses_custom_tools(request: Mapping[str, object]) -> Mapping[str, object]:
    """Adapt custom Responses tools and history for native function-only providers."""
    custom_tool_names: Final = native_responses_custom_tool_name_map(request)
    namespace_tool_names: Final = native_responses_namespace_tool_name_map(request)
    request_tools: Final = _tools_from_request(request)
    if request_tools is not None:
        _validate_native_responses_namespace_tools(request_tools)
    namespace_descriptions: Final = tuple(
        description
        for tool in request_tools or ()
        for description in (_native_namespace_description(tool, custom_tool_names),)
        if description is not None
    )
    instructions: Final = request.get("instructions")
    if namespace_descriptions and instructions is not None and not isinstance(instructions, str):
        raise ValueError("Responses instructions must be a string")
    normalized_tools: Final = (
        _normalize_native_responses_tools(request_tools, custom_tool_names)
        if request_tools is not None
        else request.get("tools")
    )
    raw_input: Final = _json_object_sequence(request.get("input"))
    normalized_input: Final = (
        tuple(
            _normalize_native_responses_input_item(item, custom_tool_names, namespace_tool_names) for item in raw_input
        )
        if raw_input is not None
        else request.get("input")
    )
    normalized_request: Final = _readonly_mapping(
        tuple(request.items())
        + (
            (
                (
                    "instructions",
                    "\n\n".join(
                        ((instructions,) if isinstance(instructions, str) and instructions else ())
                        + namespace_descriptions
                    ),
                ),
            )
            if namespace_descriptions
            else ()
        )
        + (("tools", normalized_tools),) * ("tools" in request)
        + (("input", normalized_input),) * ("input" in request)
        + (
            (
                (
                    "tool_choice",
                    _normalize_native_responses_tool_choice(
                        request["tool_choice"], custom_tool_names, namespace_tool_names
                    ),
                ),
            )
            if "tool_choice" in request
            else ()
        )
    )
    return _json_request_payload(normalized_request)


def _validate_native_responses_namespace_tools(tools: Sequence[Mapping[str, object]]) -> None:
    from .transformation import LiteLLMCompletionResponsesConfig

    LiteLLMCompletionResponsesConfig.responses_tools_to_chat_forms(tools)


def _wire_name_for_namespace_function(
    name: object,
    namespace: object,
    namespace_tool_names: Mapping[str, tuple[str, str]],
) -> str | None:
    if not isinstance(name, str) or not isinstance(namespace, str):
        return None
    mapped_wire_name: Final = next(
        (wire_name for wire_name, identity in namespace_tool_names.items() if identity == (namespace, name)),
        None,
    )
    return mapped_wire_name or _wire_name_without_definition(name, namespace)


def _wire_name_without_definition(name: str, namespace: str | None) -> str:
    return f"{namespace}__{name}" if namespace else name


def build_tool_call_item_kwargs(
    call_id: str,
    name: str,
    arguments_or_input: str,
    status: str,
    custom_tool_names: set[str],
) -> dict[str, str]:
    """Build kwargs for an output item dict that is either a ``function_call``
    or a ``custom_tool_call`` depending on whether *name* is in
    *custom_tool_names*.

    For custom tools the ``arguments`` JSON is unwrapped into the ``input``
    field. For regular function tools the raw ``arguments`` string is kept.

    This centralises the branching logic so the streaming iterator and the
    non-streaming transformation share a single code path.
    """
    custom: Final = is_custom_tool_call(name, custom_tool_names)
    item_type: Final = "custom_tool_call" if custom else "function_call"
    kwargs: Final[dict[str, str]] = {
        "type": item_type,
        "id": openai_shaped_tool_call_item_id(item_type, call_id),
        "call_id": call_id,
        "name": name,
        "status": status,
    }
    if custom:
        if status == "completed":
            kwargs["input"] = unwrap_custom_tool_arguments(arguments_or_input)
        else:
            kwargs["input"] = ""
    else:
        kwargs["arguments"] = arguments_or_input
    return kwargs


class _CustomToolFormat(BaseModel):
    syntax: str = ""
    definition: str = ""


_ALLOWED_CALLERS_ADAPTER: Final[TypeAdapter[list[str] | None]] = TypeAdapter(list[str] | None)


def validated_allowed_callers(value: object) -> list[str] | None:
    try:
        return _ALLOWED_CALLERS_ADAPTER.validate_python(value, strict=True)
    except ValidationError as exc:
        raise ValueError("allowed_callers must be a list of strings") from exc


def custom_tool_grammar_suffix(fmt: object) -> str:
    try:
        parsed: Final = _CustomToolFormat.model_validate(fmt)
    except ValidationError:
        return ""
    if not parsed.definition:
        return ""
    return f"\n\nFormat:\n```{parsed.syntax}\n{parsed.definition}\n```"


def convert_custom_tool_to_function_tool(tool: Mapping[str, object]) -> ChatCompletionToolParam | None:
    """Convert a Responses API ``custom`` tool to a Chat Completions ``function``
    tool.

    The original description and grammar apply to the content string parameter.
    Returns ``None`` if the tool is not a custom tool. Raises ``ValueError`` if
    ``allowed_callers`` is not a list of strings.
    """
    if tool.get("type") != "custom":
        return None
    raw_name: Final = tool.get("name")
    name: Final = raw_name if isinstance(raw_name, str) else ""
    raw_description: Final = tool.get("description")
    content_description: Final = (
        raw_description if isinstance(raw_description, str) else ""
    ) + custom_tool_grammar_suffix(tool.get("format"))
    allowed_callers: Final = validated_allowed_callers(tool.get("allowed_callers"))
    function_chunk: Final = ChatCompletionToolParamFunctionChunk(
        name=name,
        description=(
            f"Call {name} with a JSON object containing the required content string. "
            "The content field holds the complete tool input. Its description and grammar apply only "
            "inside that string, not to the outer JSON arguments."
        ),
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": content_description or f"The complete input for {name}.",
                }
            },
            "required": ["content"],
        },
    )
    if allowed_callers is None:
        return ChatCompletionToolParam(type="function", function=function_chunk)
    return ChatCompletionToolParam(type="function", function=function_chunk, allowed_callers=allowed_callers)
