import asyncio
import json
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import REDACTED_BY_LITELLM, REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER
from litellm.litellm_core_utils.cold_storage_handler import ColdStorageHandler
from litellm.litellm_core_utils.core_helpers import as_str_mapping, attribute_or_none, require_attributes
from litellm.responses.litellm_completion_transformation.client_tool_search import (
    bridge_client_tool_search,
    client_tool_search_loaded_tools,
)
from litellm.responses.litellm_completion_transformation.compaction_items import (
    holds_compaction_summary,
    input_items,
    instruction_items,
    response_holds_compaction_block,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionResponseMessage,
    GenericChatCompletionMessage,
    ResponseInputParam,
)
from litellm.types.utils import CallTypes, ChatCompletionMessageToolCall, Message, ModelResponse

if TYPE_CHECKING:
    from litellm.proxy._types import SpendLogsMetadata, SpendLogsPayload
    from litellm.responses.litellm_completion_transformation.transformation import (
        ChatCompletionSession,
        SessionMessage,
    )
else:
    ChatCompletionSession = Any

########################################################
# Cold Storage Handler
########################################################
COLD_STORAGE_HANDLER: Final = ColdStorageHandler()
########################################################


class _CarriesArguments(Protocol):
    arguments: object


def _normalize_redacted_function_arguments(tool_call: object) -> None:
    function: Final = attribute_or_none(tool_call, "function")
    if function is None:
        return
    carrier: Final = require_attributes(function, _CarriesArguments, "arguments")
    if carrier.arguments == REDACTED_BY_LITELLM:
        carrier.arguments = REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER


def _normalize_redacted_tool_call_arguments(message: Message) -> None:
    """Redaction stores the bare sentinel (invalid JSON) in tool-call arguments;
    normalize replayed history to "{}" so provider converters can parse it."""
    for tool_call in message.tool_calls or []:
        _normalize_redacted_function_arguments(tool_call)
    function_call: Final = message.function_call
    if function_call is not None and function_call.arguments == REDACTED_BY_LITELLM:
        function_call.arguments = REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER


_MCP_TOOL_CALL_TYPES: Final = frozenset((CallTypes.list_mcp_tools.value, CallTypes.call_mcp_tool.value))


_FAILED_TURN_STATUS: Final = "failure"


def _is_a_failed_turn(spend_log: "SpendLogsPayload") -> bool:
    return "status" in spend_log and spend_log["status"] == _FAILED_TURN_STATUS


def _is_a_conversation_turn(spend_log: "SpendLogsPayload") -> bool:
    return spend_log.get("call_type") not in _MCP_TOOL_CALL_TYPES and not _is_a_failed_turn(spend_log)


_RESPONSE_READ_CALL_TYPES: Final = frozenset(("aget_responses", "acancel_responses"))


def _output_item_id(item: object) -> str | None:
    match item:
        case {"id": str() as item_id}:
            return item_id
        case _:
            return None


def _is_a_response_read(spend_log: "SpendLogsPayload") -> bool:
    return "call_type" in spend_log and spend_log["call_type"] in _RESPONSE_READ_CALL_TYPES


def _stored_response(spend_log: "SpendLogsPayload") -> object:
    return spend_log.get("response")


def _output_items(spend_log: "SpendLogsPayload") -> Iterator[tuple[str | None, object]]:
    output: Final = _stored_responses_api_output(_stored_response(spend_log)) or []
    return zip(map(_output_item_id, output), output)


def _identified_output_items(spend_log: "SpendLogsPayload") -> Iterator[tuple[str, object]]:
    return ((item_id, item) for item_id, item in _output_items(spend_log) if item_id is not None)


def _output_item_ids(spend_log: "SpendLogsPayload") -> frozenset[str]:
    return frozenset(item_id for item_id, _ in _identified_output_items(spend_log))


def _first_index_holding_each_item(indexed_turns: Iterable[tuple[int, "SpendLogsPayload"]]) -> Mapping[str, int]:
    return dict(
        chain.from_iterable(
            ((item_id, index) for item_id in _output_item_ids(turn)) for index, turn in reversed(tuple(indexed_turns))
        )
    )


def _turn_replaying_each_item(turns: Sequence["SpendLogsPayload"]) -> Mapping[str, int]:
    indexed_turns: Final = tuple(enumerate(turns))
    first_read_holding: Final = _first_index_holding_each_item(
        (index, turn) for index, turn in indexed_turns if _is_a_response_read(turn)
    )
    first_other_turn_holding: Final = _first_index_holding_each_item(
        (index, turn) for index, turn in indexed_turns if not _is_a_response_read(turn)
    )
    return MappingProxyType({**first_read_holding, **first_other_turn_holding})


def _latest_read_copy_of_each_item(turns: Sequence["SpendLogsPayload"]) -> Mapping[str, object]:
    return MappingProxyType(
        dict(chain.from_iterable(_identified_output_items(turn) for turn in turns if _is_a_response_read(turn)))
    )


def _with_only_new_output(
    index: int, turn: "SpendLogsPayload", turn_replaying: Mapping[str, int], latest_copy: Mapping[str, object]
) -> "SpendLogsPayload | None":
    if not _is_a_response_read(turn):
        return turn
    stored_response: Final = as_str_mapping(_stored_response(turn))
    new_output: Final = [
        item if item_id is None else latest_copy[item_id]
        for item_id, item in _output_items(turn)
        if item_id is None or turn_replaying.get(item_id) == index
    ]
    if stored_response is None or not new_output:
        return None
    replayed_turn: Final[SpendLogsPayload] = {**turn, "response": {**stored_response, "output": new_output}}
    return replayed_turn


def _conversation_turns(spend_logs: Sequence["SpendLogsPayload"]) -> tuple["SpendLogsPayload", ...]:
    turns: Final = tuple(filter(_is_a_conversation_turn, spend_logs))
    turn_replaying: Final = _turn_replaying_each_item(turns)
    latest_copy: Final = _latest_read_copy_of_each_item(turns)
    replayed: Final = (
        _with_only_new_output(index, turn, turn_replaying, latest_copy) for index, turn in enumerate(turns)
    )
    return tuple(turn for turn in replayed if turn is not None)


_MCP_AUTO_EXECUTE_OUTPUT_TYPES: Final = frozenset(("mcp_tools_fetched", "tool_execution_results"))


def _is_model_output(item: object) -> bool:
    return not (isinstance(item, Mapping) and item.get("type") in _MCP_AUTO_EXECUTE_OUTPUT_TYPES)


def _stored_responses_api_output(stored_response: object) -> ResponseInputParam | None:
    match stored_response:
        case {"object": "response", "output": list() as output}:
            return [item for item in output if _is_model_output(item)]
        case {"object": "response"}:
            return []
        case _:
            return None


@dataclass(frozen=True, slots=True)
class _ReplayStart:
    index: int
    at_output: bool


@dataclass(frozen=True, slots=True)
class StoredTurns:
    spend_logs: tuple["SpendLogsPayload", ...]
    proxy_server_requests: tuple[Mapping[str, object] | None, ...]


def _same_spend_log_owner(first: Mapping[str, object], second: Mapping[str, object]) -> bool:
    user: Final = first.get("user")
    team_id: Final = first.get("team_id")
    if user != second.get("user") or team_id != second.get("team_id"):
        return False
    api_key: Final = first.get("api_key")
    return bool(user or team_id) or (api_key is not None and api_key == second.get("api_key"))


def stored_request_id(row: Mapping[str, object]) -> str | None:
    request_id: Final = row.get("request_id")
    return request_id if isinstance(request_id, str) else None


def stored_session_id(row: Mapping[str, object]) -> str | None:
    session_id: Final = row.get("session_id")
    return session_id if isinstance(session_id, str) else None


def _stored_request_ids(turns: StoredTurns) -> tuple[str, ...]:
    return tuple(request_id for row in turns.spend_logs if (request_id := stored_request_id(row)) is not None)


def _previous_stored_response(request: Mapping[str, object] | None) -> str | None:
    previous: Final = request.get("previous_response_id") if request is not None else None
    return previous if isinstance(previous, str) and previous else None


def _previous_stored_responses(turns: StoredTurns) -> tuple[str, ...]:
    return tuple(
        previous
        for request in turns.proxy_server_requests
        if (previous := _previous_stored_response(request)) is not None
    )


def _normalized_end_time(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if not isinstance(value, str):
        return None
    try:
        return _normalized_end_time(
            datetime.fromisoformat(value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value)
        )
    except ValueError:
        return None


def _stored_end_time(row: Mapping[str, object]) -> tuple[int, datetime]:
    end_time: Final = _normalized_end_time(row.get("endTime"))
    return (0, end_time) if end_time is not None else (1, datetime.max.replace(tzinfo=timezone.utc))


def _ordered_stored_turns(batches: Sequence[StoredTurns]) -> StoredTurns:
    if len(batches) == 1:
        return batches[0]
    pairs: Final = tuple(
        chain.from_iterable(zip(batch.spend_logs, batch.proxy_server_requests, strict=True) for batch in batches)
    )
    unique: Final = {
        (stored_request_id(row), index if stored_request_id(row) is None else 0): (row, request)
        for index, (row, request) in enumerate(pairs)
    }
    ordered: Final = sorted(unique.values(), key=lambda pair: _stored_end_time(pair[0]))
    return StoredTurns(
        spend_logs=tuple(row for row, _ in ordered),
        proxy_server_requests=tuple(request for _, request in ordered),
    )


def _compaction_point(
    index: int, spend_log: "SpendLogsPayload", proxy_server_request: Mapping[str, object] | None
) -> _ReplayStart | None:
    if response_holds_compaction_block(_stored_response(spend_log)):
        return _ReplayStart(index=index, at_output=True)
    if proxy_server_request is not None and holds_compaction_summary(proxy_server_request.get("input")):
        return _ReplayStart(index=index, at_output=False)
    return None


def _replay_start(
    spend_logs: Sequence["SpendLogsPayload"], proxy_server_requests: Sequence[Mapping[str, object] | None]
) -> _ReplayStart:
    return next(
        (
            start
            for index in range(len(spend_logs) - 1, -1, -1)
            if (start := _compaction_point(index, spend_logs[index], proxy_server_requests[index])) is not None
        ),
        _ReplayStart(index=0, at_output=False),
    )


def _holds_compaction_point(
    spend_logs: Sequence["SpendLogsPayload"], proxy_server_requests: Sequence[Mapping[str, object] | None]
) -> bool:
    return any(
        _compaction_point(index, spend_log, proxy_server_request) is not None
        for index, (spend_log, proxy_server_request) in enumerate(zip(spend_logs, proxy_server_requests, strict=True))
    )


def referenced_request_ids(previous_response_id: str, client_response_ids: tuple[str, ...]) -> tuple[str, ...]:
    decoded_response_id: Final = ResponsesAPIRequestUtils.decode_responses_api_response_id(previous_response_id)
    response_id: Final = decoded_response_id.get("response_id", previous_response_id)
    return tuple(dict.fromkeys((response_id, previous_response_id, *client_response_ids)))


def _turns_through_referenced_response(
    spend_logs: Sequence["SpendLogsPayload"], referenced_request_ids: tuple[str, ...]
) -> int:
    return next(
        (
            index + 1
            for index in range(len(spend_logs) - 1, -1, -1)
            if spend_logs[index].get("request_id") in referenced_request_ids
        ),
        len(spend_logs),
    )


def _replayed_request(
    index: int,
    proxy_server_request: Mapping[str, object] | None,
    start: _ReplayStart,
    earlier_instructions: tuple[object, ...],
) -> Mapping[str, object] | None:
    if proxy_server_request is None or index != start.index:
        return proxy_server_request
    stored_input: Final = proxy_server_request.get("input")
    if start.at_output:
        return {
            **proxy_server_request,
            "input": [*earlier_instructions, *instruction_items(stored_input)],
            "messages": None,
        }
    if not earlier_instructions:
        return proxy_server_request
    return {**proxy_server_request, "input": [*earlier_instructions, *input_items(stored_input)]}


def _instruction_items_before(
    proxy_server_requests: Sequence[Mapping[str, object] | None], start: _ReplayStart
) -> tuple[object, ...]:
    return tuple(
        chain.from_iterable(
            instruction_items(request.get("input"))
            for request in proxy_server_requests[: start.index]
            if request is not None
        )
    )


def _stored_request_tools(proxy_server_request: Mapping[str, object] | None) -> object | None:
    """The stored turn's ``tools``, so its calls read the same namespace shapes it was sent with."""
    return proxy_server_request.get("tools") if proxy_server_request is not None else None


def _stored_instructions(proxy_server_request: Mapping[str, object] | None) -> str | None:
    instructions: Final = None if proxy_server_request is None else proxy_server_request.get("instructions")
    return instructions if isinstance(instructions, str) and instructions else None


def _latest_replayable_instructions(proxy_server_requests: Sequence[Mapping[str, object] | None]) -> str | None:
    latest: Final = next(
        (instructions for instructions in map(_stored_instructions, reversed(proxy_server_requests)) if instructions),
        None,
    )
    return None if latest == REDACTED_BY_LITELLM else latest


class ResponsesSessionHandler:
    @staticmethod
    async def get_chat_completion_message_history_for_previous_response_id(
        previous_response_id: str,
        item_cache_control: bool = False,
        client_response_ids: tuple[str, ...] = (),
        tool_call_provider_fields: bool = False,
        stored_turns: StoredTurns | None = None,
    ) -> ChatCompletionSession:
        """
        Return the chat completion message history for a previous response id
        """
        from litellm.responses.litellm_completion_transformation.transformation import (
            ChatCompletionSession,
            with_stored_calls_the_input_answers,
        )

        verbose_proxy_logger.debug("inside get_chat_completion_message_history_for_previous_response_id")
        turns: Final = (
            stored_turns
            if stored_turns is not None
            else await ResponsesSessionHandler.get_stored_turns_for_previous_response_id(
                previous_response_id, client_response_ids
            )
        )
        all_spend_logs: Final = turns.spend_logs
        verbose_proxy_logger.debug("found %s spend logs for this response id", len(all_spend_logs))

        referenced_ids: Final = referenced_request_ids(previous_response_id, client_response_ids)
        referenced_log: Final = next(
            (row for row in reversed(all_spend_logs) if stored_request_id(row) in referenced_ids),
            all_spend_logs[0] if all_spend_logs else None,
        )
        litellm_session_id: Final = stored_session_id(referenced_log) if referenced_log is not None else None
        proxy_server_requests: Final = turns.proxy_server_requests
        replayable_turns: Final = _turns_through_referenced_response(all_spend_logs, referenced_ids)
        start: Final = _replay_start(all_spend_logs[:replayable_turns], proxy_server_requests[:replayable_turns])
        replayed_turns: Final = (
            replayable_turns if _holds_compaction_point(all_spend_logs, proxy_server_requests) else len(all_spend_logs)
        )
        earlier_instructions: Final = _instruction_items_before(proxy_server_requests, start)
        replayed_history: Final = ResponsesSessionHandler._history_of_turns(
            all_spend_logs[start.index : replayed_turns],
            [
                _replayed_request(index, proxy_server_requests[index], start, earlier_instructions)
                for index in range(start.index, replayed_turns)
            ],
            item_cache_control,
            tool_call_provider_fields,
        )
        history_before_replay: Final = ResponsesSessionHandler._history_of_turns(
            all_spend_logs[: start.index],
            proxy_server_requests[: start.index],
            item_cache_control,
            tool_call_provider_fields,
        )
        chat_completion_message_history: Final = (
            list(with_stored_calls_the_input_answers(replayed_history, history_before_replay))
            if history_before_replay
            else replayed_history
        )

        verbose_proxy_logger.debug(
            "chat_completion_message_history %s",
            json.dumps(chat_completion_message_history, indent=4, default=str),
        )
        return ChatCompletionSession(
            messages=chat_completion_message_history,
            litellm_session_id=litellm_session_id,
            instructions=_latest_replayable_instructions(proxy_server_requests),
        )

    @staticmethod
    async def get_stored_turns_for_previous_response_id(
        previous_response_id: str,
        client_response_ids: tuple[str, ...] = (),
    ) -> StoredTurns:
        current: Final = await ResponsesSessionHandler._load_stored_turns(previous_response_id, client_response_ids)
        if not current.spend_logs:
            return current

        anchor_ids: Final = referenced_request_ids(previous_response_id, client_response_ids)
        anchor: Final = next(
            (row for row in current.spend_logs if stored_request_id(row) in anchor_ids), current.spend_logs[-1]
        )
        batches: Final[list[StoredTurns]] = [current]  # mutable-ok: append independently loaded ancestor groups
        visited: Final[set[str]] = set(_stored_request_ids(current))  # mutable-ok: bound cycles in stored links
        pending: Final[deque[str]] = deque(_previous_stored_responses(current))  # mutable-ok: queue ancestor lookups
        previous: str = ""  # rebind-ok: process each pending response id
        earlier: StoredTurns | None = None  # rebind-ok: successive ancestors arrive asynchronously
        while pending:
            previous = pending.popleft()
            earlier = await ResponsesSessionHandler._earlier_stored_turns(previous, visited, anchor)
            visited.update(referenced_request_ids(previous, ()))
            if earlier is None:
                continue
            batches.append(earlier)
            visited.update(_stored_request_ids(earlier))
            pending.extend(_previous_stored_responses(earlier))

        return _ordered_stored_turns(batches)

    @staticmethod
    async def _earlier_stored_turns(
        previous_response_id: str, visited: Set[str], owner: Mapping[str, object]
    ) -> StoredTurns | None:
        referenced: Final = referenced_request_ids(previous_response_id, ())
        if any(request_id in visited for request_id in referenced):
            return None
        prior: Final = await ResponsesSessionHandler._load_stored_turns(previous_response_id, (), owner=owner)
        if not any(stored_request_id(row) in referenced for row in prior.spend_logs):
            return None
        end: Final = _turns_through_referenced_response(prior.spend_logs, referenced)
        return StoredTurns(prior.spend_logs[:end], prior.proxy_server_requests[:end])

    @staticmethod
    async def _load_stored_turns(
        previous_response_id: str,
        client_response_ids: tuple[str, ...],
        *,
        owner: Mapping[str, object] | None = None,
    ) -> StoredTurns:
        spend_logs: Final = _conversation_turns(
            tuple(
                spend_log
                for spend_log in await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(
                    previous_response_id, client_response_ids
                )
                if owner is None or _same_spend_log_owner(owner, spend_log)
            )
        )
        return StoredTurns(
            spend_logs=spend_logs,
            proxy_server_requests=tuple(
                [
                    await ResponsesSessionHandler.get_proxy_server_request_from_spend_log(spend_log=spend_log)
                    for spend_log in spend_logs
                ]
            ),
        )

    @staticmethod
    def client_tool_search_tools_loaded_before(
        stored_turns: StoredTurns,
        previous_response_id: str,
        client_response_ids: tuple[str, ...] = (),
    ) -> tuple[Mapping[str, object], ...]:
        turns: Final = _turns_through_referenced_response(
            stored_turns.spend_logs, referenced_request_ids(previous_response_id, client_response_ids)
        )
        return tuple(
            chain.from_iterable(
                client_tool_search_loaded_tools(request.get("input"))
                for request in stored_turns.proxy_server_requests[:turns]
                if request is not None
            )
        )

    @staticmethod
    def _history_of_turns(
        spend_logs: Sequence["SpendLogsPayload"],
        proxy_server_requests: Sequence[Mapping[str, object] | None],
        item_cache_control: bool,
        tool_call_provider_fields: bool,
    ) -> list["SessionMessage"]:
        return list(
            chain.from_iterable(
                ResponsesSessionHandler.extend_chat_completion_message_with_spend_log_payload(
                    spend_log=spend_log,
                    proxy_server_request_dict=proxy_server_request,
                    chat_completion_message_history=[],
                    item_cache_control=item_cache_control,
                    tool_call_provider_fields=tool_call_provider_fields,
                )
                for spend_log, proxy_server_request in zip(spend_logs, proxy_server_requests, strict=True)
            )
        )

    @staticmethod
    def extend_chat_completion_message_with_spend_log_payload(
        spend_log: "SpendLogsPayload",
        proxy_server_request_dict: Mapping[str, object] | None,
        chat_completion_message_history: list[
            AllMessageValues
            | GenericChatCompletionMessage
            | ChatCompletionMessageToolCall
            | ChatCompletionResponseMessage
            | Message
        ],
        item_cache_control: bool = False,
        tool_call_provider_fields: bool = False,
    ):
        """
        Extend the chat completion message history with the spend log payload
        """
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        response_input_param: str | ResponseInputParam | None = None
        stored_tools: Final = _stored_request_tools(proxy_server_request_dict)

        ############################################################
        # Add Input messages for this Spend Log
        ############################################################
        if proxy_server_request_dict:
            _response_input_param: Final = proxy_server_request_dict.get("input") or proxy_server_request_dict.get(
                "messages"
            )
            if isinstance(_response_input_param, (str, list)):
                response_input_param = _response_input_param
            elif isinstance(_response_input_param, dict):
                response_input_param = cast(
                    ResponseInputParam,
                    [_response_input_param],
                )

        if response_input_param:
            chat_completion_message_history.extend(
                LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                    input=bridge_client_tool_search(response_input_param, None).input,
                    responses_api_request={"tools": stored_tools} if stored_tools is not None else {},
                    replay_reasoning=True,
                    item_cache_control=item_cache_control,
                    tool_call_provider_fields=tool_call_provider_fields,
                    only_since_compaction=True,
                    replay_tool_search=True,
                )
            )

        ############################################################
        # Add Output messages for this Spend Log
        ############################################################
        _response_output: Final = _stored_response(spend_log)
        if (responses_api_output := _stored_responses_api_output(_response_output)) is not None:
            chat_completion_message_history.extend(
                LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                    input=bridge_client_tool_search(responses_api_output, None).input,
                    responses_api_request={"tools": stored_tools} if stored_tools is not None else {},
                    replay_reasoning=True,
                    item_cache_control=item_cache_control,
                    tool_call_provider_fields=tool_call_provider_fields,
                    only_since_compaction=True,
                    replay_tool_search=True,
                )
            )
        elif isinstance(_response_output, dict) and _response_output and _response_output != {}:
            # transform `ChatCompletion Response` to `ResponsesAPIResponse`
            model_response: Final = ModelResponse(**_response_output)
            for choice in model_response.choices:
                if hasattr(choice, "message"):
                    _normalize_redacted_tool_call_arguments(choice.message)
                    chat_completion_message_history.append(choice.message)
        return chat_completion_message_history

    @staticmethod
    async def get_proxy_server_request_from_spend_log(
        spend_log: "SpendLogsPayload",
    ) -> dict[str, object] | None:
        """
        Get the parsed proxy server request from the spend log
        """
        proxy_server_request: Final[str | dict] = spend_log.get("proxy_server_request") or "{}"
        proxy_server_request_dict: dict[str, object] | None = None
        if isinstance(proxy_server_request, dict):
            proxy_server_request_dict = proxy_server_request
        else:
            proxy_server_request_dict = json.loads(proxy_server_request)

        ############################################################
        # Check if user has setup cold storage for session handling
        ############################################################
        if ResponsesSessionHandler._should_check_cold_storage_for_full_payload(proxy_server_request_dict):
            # Try to get cold storage object key from spend log metadata
            _proxy_server_request_dict: dict | None = None
            cold_storage_object_key = ResponsesSessionHandler._get_cold_storage_object_key_from_spend_log(spend_log)
            if cold_storage_object_key:
                # Use the object key directly from metadata
                _proxy_server_request_dict = (
                    await ResponsesSessionHandler.get_proxy_server_request_from_cold_storage_with_object_key(
                        object_key=cold_storage_object_key,
                    )
                )
            if _proxy_server_request_dict:
                proxy_server_request_dict = _proxy_server_request_dict

        return proxy_server_request_dict

    @staticmethod
    def _get_cold_storage_object_key_from_spend_log(
        spend_log: "SpendLogsPayload",
    ) -> str | None:
        """
        Extract the cold storage object key from spend log metadata.

        Args:
            spend_log: The spend log payload containing metadata

        Returns:
            Optional[str]: The cold storage object key if found, None otherwise
        """
        try:
            metadata_str: Final = spend_log.get("metadata", "{}")
            if isinstance(metadata_str, str):
                metadata_dict: Final[SpendLogsMetadata] = json.loads(metadata_str)
                return metadata_dict.get("cold_storage_object_key")
            elif isinstance(metadata_str, dict):
                return metadata_str.get("cold_storage_object_key")
            return None
        except (json.JSONDecodeError, TypeError, AttributeError):
            verbose_proxy_logger.debug("Failed to parse metadata from spend log to extract cold storage object key")
            return None

    @staticmethod
    async def get_proxy_server_request_from_cold_storage_with_object_key(
        object_key: str,
    ) -> dict[str, object] | None:
        """
        Get the proxy server request from cold storage using the object key directly.

        Args:
            object_key: The S3/GCS object key to retrieve

        Returns:
            Optional[dict]: The proxy server request dict or None if not found
        """
        verbose_proxy_logger.debug("inside get_proxy_server_request_from_cold_storage_with_object_key...")

        proxy_server_request_dict: Final = (
            await COLD_STORAGE_HANDLER.get_proxy_server_request_from_cold_storage_with_object_key(
                object_key=object_key,
            )
        )

        return proxy_server_request_dict

    @staticmethod
    def _should_check_cold_storage_for_full_payload(
        proxy_server_request_dict: Mapping[str, object] | None,
    ) -> bool:
        """
        Only check cold storage when both are true
        1. `LITELLM_TRUNCATED_PAYLOAD_FIELD` is in the proxy server request dict
        2. `litellm.cold_storage_custom_logger` is not None
        """
        from litellm.constants import LITELLM_TRUNCATED_PAYLOAD_FIELD

        configured_cold_storage_custom_logger: Final = litellm.cold_storage_custom_logger
        if configured_cold_storage_custom_logger is None:
            return False
        if proxy_server_request_dict is None:
            return True
        if len(proxy_server_request_dict) == 0:
            return True
        if LITELLM_TRUNCATED_PAYLOAD_FIELD in str(proxy_server_request_dict):
            return True
        return False

    @staticmethod
    async def get_all_spend_logs_for_previous_response_id(
        previous_response_id: str,
        client_response_ids: tuple[str, ...] = (),
        *,
        include_session_history: bool = True,
    ) -> "list[SpendLogsPayload]":
        """
        Get all spend logs for a previous response id


        SQL query

        SELECT session_id, api_key, "user", team_id FROM spend_logs WHERE request_id = previous_response_id, then
        SELECT * FROM spend_logs with that session_id and the same owner

        Only rows with the referenced response's owner are returned: the same user and team, and for a key with
        neither, the same key. The session id can be caller supplied, so matching on it alone would mix owners. A
        user's or team's regenerated or second key keeps the history; a key with neither does not, and turns written
        by another owner (an admin continuing the chain) are not replayed.

        A just-finished turn gets a short second chance: the worker that served it may
        still be writing its spend log when the follow-up arrives, and an empty result
        drops the whole conversation instead of erroring. Deployments that write no spend
        logs at all have nothing to wait for, so they keep the single original query.
        """
        from litellm.constants import (
            RESPONSES_SESSION_LOOKUP_MAX_ATTEMPTS,
            RESPONSES_SESSION_LOOKUP_RETRY_INTERVAL,
        )
        from litellm.proxy.proxy_server import disable_spend_logs, prisma_client

        verbose_proxy_logger.debug("decoding response id=%s", previous_response_id)

        request_ids: Final = list(referenced_request_ids(previous_response_id, client_response_ids))
        response_id: Final = request_ids[0]
        if prisma_client is None:
            return []

        matching_rows: Final = (
            "matching_session.session_id = logs.session_id "
            "OR (matching_session.session_id IS NULL AND matching_session.request_id = logs.request_id)"
            if include_session_history
            else "matching_session.request_id = logs.request_id"
        )
        query: Final = f"""
            WITH matching_session AS (
                SELECT request_id, session_id, api_key, "user", team_id
                FROM "LiteLLM_SpendLogs"
                WHERE request_id = ANY($1::text[])
            )
            SELECT *
            FROM "LiteLLM_SpendLogs" AS logs
            WHERE EXISTS (
                SELECT 1
                FROM matching_session
                WHERE ({matching_rows})
                  AND (
                      matching_session."user" = logs."user"
                      OR (matching_session."user" IS NULL AND logs."user" IS NULL)
                  )
                  AND (
                      matching_session.team_id = logs.team_id
                      OR (matching_session.team_id IS NULL AND logs.team_id IS NULL)
                  )
                  AND (
                      COALESCE(matching_session."user", '') <> ''
                      OR COALESCE(matching_session.team_id, '') <> ''
                      OR (
                          COALESCE(matching_session.api_key, '') <> ''
                          AND matching_session.api_key = logs.api_key
                      )
                  )
            )
            ORDER BY "endTime" ASC;
        """

        max_attempts: Final = 1 if disable_spend_logs else RESPONSES_SESSION_LOOKUP_MAX_ATTEMPTS
        for attempt in range(max_attempts):
            if attempt:
                await asyncio.sleep(RESPONSES_SESSION_LOOKUP_RETRY_INTERVAL)
            if spend_logs := await prisma_client.db.query_raw(query, request_ids):
                verbose_proxy_logger.debug(
                    "Found the following spend logs for previous response id %s: %s",
                    response_id,
                    json.dumps(spend_logs, indent=4, default=str),
                )
                return spend_logs

        verbose_proxy_logger.debug("Found no spend logs for previous response id %s", response_id)
        return []
