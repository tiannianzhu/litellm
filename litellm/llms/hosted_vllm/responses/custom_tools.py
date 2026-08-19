import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm.responses.litellm_completion_transformation.custom_tools import (
    MAX_CUSTOM_TOOL_ARGUMENTS_LEN,
    unwrap_custom_tool_arguments_strict,
)

_OBJECT: Final = TypeAdapter(dict[str, object])
_OUTPUT: Final = TypeAdapter(list[dict[str, object]])


@dataclass(frozen=True, slots=True)
class _CustomCall:
    output_index: int
    name: str
    call_id: str
    fragments: tuple[str, ...] = ()
    size: int = 0
    input: str | None = None
    item_done: bool = False


class NativeResponsesCustomToolAdapter:
    def __init__(
        self, names: Mapping[str, tuple[str, str | None]], namespace_names: Mapping[str, tuple[str, str]] | None = None
    ) -> None:
        self.names = names
        self.namespace_names = namespace_names or MappingProxyType({})
        self._calls: Mapping[str, _CustomCall] = MappingProxyType({})
        self._sequence_number = 0

    def item(self, item: Mapping[str, object], *, partial: bool = False) -> Mapping[str, object]:
        name: Final = item.get("name")
        if item.get("type") != "function_call" or not isinstance(name, str):
            return item
        if name not in self.names:
            if name in self.namespace_names:
                function_namespace, function_name = self.namespace_names[name]
                return MappingProxyType({**item, "name": function_name, "namespace": function_namespace})
            return item
        original_name, namespace = self.names[name]
        return MappingProxyType(
            {
                **MappingProxyType(
                    {key: value for key, value in item.items() if key not in ("arguments", "name", "type")}
                ),
                "type": "custom_tool_call",
                "name": original_name,
                **(MappingProxyType({"namespace": namespace}) if namespace is not None else MappingProxyType({})),
                "input": "" if partial else unwrap_custom_tool_arguments_strict(item.get("arguments")),
            }
        )

    def response(self, response: Mapping[str, object]) -> Mapping[str, object]:
        output: Final = response.get("output")
        if output is None:
            return response
        partial: Final = response.get("status") in ("failed", "incomplete", "in_progress", "queued", "cancelled")
        return MappingProxyType(
            {
                **response,
                "output": tuple(self.item(item, partial=partial) for item in _OUTPUT.validate_python(output)),
            }
        )

    def _finish_input(self, event: Mapping[str, object], arguments: object) -> tuple[Mapping[str, object], ...]:
        item_id: Final = event.get("item_id")
        if not isinstance(item_id, str) or item_id not in self._calls:
            raise ValueError("Custom tool arguments are missing their output item")
        call: Final = self._calls[item_id]
        content: Final = unwrap_custom_tool_arguments_strict(arguments)
        if call.input is not None:
            if content != call.input:
                raise ValueError("Custom tool input differs between completion events")
            return ()
        self._calls = MappingProxyType({**self._calls, item_id: replace(call, input=content, fragments=())})
        fields: Final = MappingProxyType({"item_id": item_id, "output_index": call.output_index})
        return (
            MappingProxyType({**fields, "type": "response.custom_tool_call_input.delta", "delta": content}),
            MappingProxyType({**fields, "type": "response.custom_tool_call_input.done", "input": content}),
        )

    def _events(self, event: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
        event_type: Final = event.get("type")
        if event_type in ("response.output_item.added", "response.output_item.done"):
            item: Final = _OBJECT.validate_python(event.get("item"))
            name: Final = item.get("name")
            if item.get("type") != "function_call" or not isinstance(name, str) or name not in self.names:
                return (MappingProxyType({**event, "item": self.item(item)}),)
            added_item_id: Final = item.get("id")
            output_index: Final = event.get("output_index")
            call_id: Final = item.get("call_id")
            if (
                not isinstance(added_item_id, str)
                or not isinstance(output_index, int)
                or output_index < 0
                or not isinstance(call_id, str)
            ):
                raise ValueError("Custom tool output is missing its item ID or output index")
            if event_type == "response.output_item.added":
                if added_item_id in self._calls or any(
                    call.output_index == output_index for call in self._calls.values()
                ):
                    raise ValueError("Custom tool output item ID was reused")
                self._calls = MappingProxyType({**self._calls, added_item_id: _CustomCall(output_index, name, call_id)})
                return (MappingProxyType({**event, "item": self.item(item, partial=True)}),)
            original_call: Final = self._calls.get(added_item_id)
            if original_call is None or (name, call_id, output_index) != (
                original_call.name,
                original_call.call_id,
                original_call.output_index,
            ):
                raise ValueError("Custom tool output identity changed during streaming")
            input_events: Final = self._finish_input(
                MappingProxyType({"item_id": added_item_id}), item.get("arguments")
            )
            self._calls = MappingProxyType(
                {**self._calls, added_item_id: replace(self._calls[added_item_id], item_done=True)}
            )
            return (*input_events, MappingProxyType({**event, "item": self.item(item)}))
        item_id: Final = event.get("item_id")
        if isinstance(item_id, str) and item_id in self._calls:
            call: Final = self._calls[item_id]
            if event.get("output_index", call.output_index) != call.output_index:
                raise ValueError("Custom tool output index changed during streaming")
            if event_type == "response.function_call_arguments.delta":
                delta: Final = event.get("delta")
                if not isinstance(delta, str) or call.input is not None:
                    raise ValueError("Invalid custom tool arguments delta")
                if call.size + len(delta) > MAX_CUSTOM_TOOL_ARGUMENTS_LEN:
                    raise ValueError("Custom tool arguments exceed the maximum supported size")
                self._calls = MappingProxyType(
                    {
                        **self._calls,
                        item_id: replace(call, fragments=(*call.fragments, delta), size=call.size + len(delta)),
                    }
                )
                return ()
            if event_type == "response.function_call_arguments.done":
                arguments: Final = event.get("arguments", "".join(call.fragments))
                return self._finish_input(event, arguments)
        raw_response: Final = event.get("response")
        if isinstance(raw_response, Mapping):
            response: Final = _OBJECT.validate_python(raw_response)
            if event_type == "response.completed":
                return (MappingProxyType({**event, "response": self.response(self._completed_response(response))}),)
            return (MappingProxyType({**event, "response": self.response(response)}),)
        return (event,)

    def _completed_response(self, response: Mapping[str, object]) -> Mapping[str, object]:
        if any(not call.item_done for call in self._calls.values()):
            raise ValueError("Response completed before its custom tool output finished")
        output: Final = _OUTPUT.validate_python(response.get("output", ()))
        if any(call.output_index >= len(output) for call in self._calls.values()):
            raise ValueError("Completed response differs from streamed custom tool output")
        return MappingProxyType(
            {
                **response,
                "output": tuple(self._completed_item(index, item) for index, item in enumerate(output)),
            }
        )

    def _completed_item(self, index: int, item: Mapping[str, object]) -> Mapping[str, object]:
        tracked: Final = next(
            ((item_id, call) for item_id, call in self._calls.items() if call.output_index == index), None
        )
        if tracked is None:
            if item.get("type") == "function_call" and item.get("name") in self.names:
                raise ValueError("Completed response has an untracked custom tool output")
            return item
        item_id, call = tracked
        if (item.get("type"), item.get("name"), self.item(item).get("input")) != (
            "function_call",
            call.name,
            call.input,
        ):
            raise ValueError("Completed response differs from streamed custom tool output")
        return MappingProxyType({**item, "id": item_id, "call_id": call.call_id})

    def chunk(self, chunk: str) -> tuple[str, ...]:
        if not chunk or chunk == "[DONE]":
            return (chunk,)
        event: Final = _OBJECT.validate_json(chunk)
        events: Final = self._events(event)
        numbered: Final = tuple(
            (
                json.dumps(
                    MappingProxyType({**entry, "sequence_number": self._sequence_number + index}),
                    ensure_ascii=False,
                    default=_OBJECT.validate_python,
                )
                for index, entry in enumerate(events)
            )
        )
        self._sequence_number += len(events)
        return numbered
