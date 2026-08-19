import asyncio
import json
from unittest.mock import Mock

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.hosted_vllm.responses.custom_tools import NativeResponsesCustomToolAdapter
from litellm.llms.hosted_vllm.responses.transformation import HostedVLLMResponsesAPIConfig
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator, SyncResponsesAPIStreamingIterator
from litellm.types.router import GenericLiteLLMParams


def _item(name="freeform", arguments='{"content":"hello"}', item_id="fc_1", call_id="call_1"):
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _event(adapter, event):
    return [json.loads(chunk) for chunk in adapter.chunk(json.dumps(event))]


@pytest.mark.parametrize("content", ['print("hello")\n路径\\file', "😀\t\r\n", "", 'a\\"b'])
def test_custom_input_is_buffered_until_valid_and_final_output_matches(content):
    adapter = NativeResponsesCustomToolAdapter({"freeform": ("freeform", None)})
    arguments = json.dumps({"content": content})
    item = _item(arguments=arguments)
    added = _event(
        adapter,
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**item, "arguments": "", "status": "in_progress"},
        },
    )
    assert added[0]["item"] == {
        key: value
        for key, value in {**item, "type": "custom_tool_call", "input": "", "status": "in_progress"}.items()
        if key != "arguments"
    }
    for char in arguments:
        assert (
            _event(
                adapter,
                {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": char},
            )
            == []
        )
    done = _event(
        adapter,
        {"type": "response.function_call_arguments.done", "item_id": "fc_1", "output_index": 0, "arguments": arguments},
    )
    assert [e["type"] for e in done] == [
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
    ]
    assert done[0]["delta"] == done[1]["input"] == content
    item_done = _event(adapter, {"type": "response.output_item.done", "output_index": 0, "item": item})
    completed = _event(
        adapter,
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [item],
                "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            },
        },
    )
    assert item_done[0]["item"] == completed[0]["response"]["output"][0] == adapter.item(item)
    assert completed[0]["response"]["usage"]["total_tokens"] == 5
    assert [e["sequence_number"] for e in [*added, *done, *item_done, *completed]] == list(range(5))


def test_parallel_custom_and_function_calls_keep_identity_and_order():
    adapter = NativeResponsesCustomToolAdapter({"wrapped": ("same", None), "ns__custom": ("custom", "ns")})
    items = [
        _item("wrapped"),
        _item("same", '{"value":1}', "fc_2", "call_2"),
        _item("ns__custom", '{"content":"second"}', "fc_3", "call_3"),
    ]
    for index, item in enumerate(items):
        added = _event(
            adapter,
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": {**item, "arguments": "", "status": "in_progress"},
            },
        )
        assert added[0]["item"]["call_id"] == item["call_id"]
    function_delta = {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_2",
        "output_index": 1,
        "delta": '{"value":1}',
    }
    assert {k: v for k, v in _event(adapter, function_delta)[0].items() if k != "sequence_number"} == function_delta
    for index in (2, 0, 1):
        events = _event(adapter, {"type": "response.output_item.done", "output_index": index, "item": items[index]})
        assert all(e["output_index"] == index for e in events)
        assert events[-1]["item"]["id"] == items[index]["id"]
    output = adapter.response({"status": "completed", "output": items})["output"]
    assert [item["type"] for item in output] == ["custom_tool_call", "function_call", "custom_tool_call"]
    assert output[1] == items[1]
    assert output[2]["namespace"] == "ns"


@pytest.mark.parametrize("arguments", ["not json", '{"content": 12}', '{"other":"code"}', '{"content":'])
def test_malformed_custom_arguments_never_emit_executable_input(arguments):
    adapter = NativeResponsesCustomToolAdapter({"freeform": ("freeform", None)})
    _event(adapter, {"type": "response.output_item.added", "output_index": 0, "item": _item(arguments="")})
    with pytest.raises((TypeError, ValueError), match="custom tool arguments"):
        _event(adapter, {"type": "response.function_call_arguments.done", "item_id": "fc_1", "arguments": arguments})
    with pytest.raises((TypeError, ValueError), match="custom tool arguments"):
        adapter.response({"status": "completed", "output": [_item(arguments=arguments)]})


def test_failure_and_incomplete_preserve_status_without_tool_success():
    for status in ("failed", "incomplete"):
        adapter = NativeResponsesCustomToolAdapter({"freeform": ("freeform", None)})
        _event(adapter, {"type": "response.output_item.added", "output_index": 0, "item": _item(arguments="")})
        event = {
            "type": f"response.{status}",
            "response": {
                "status": status,
                "output": [_item(arguments='{"content":')],
                "error": {"message": "backend failure"},
            },
        }
        result = _event(adapter, event)
        assert len(result) == 1
        assert result[0]["type"] == event["type"]
        assert result[0]["response"]["error"] == event["response"]["error"]
        assert result[0]["response"]["output"][0]["input"] == ""


def test_inconsistent_completion_does_not_report_success():
    adapter = NativeResponsesCustomToolAdapter({"freeform": ("freeform", None)})
    _event(adapter, {"type": "response.output_item.added", "output_index": 0, "item": _item(arguments="")})
    with pytest.raises(ValueError, match="before its custom tool"):
        _event(adapter, {"type": "response.completed", "response": {"status": "completed", "output": [_item()]}})
    _event(adapter, {"type": "response.output_item.done", "output_index": 0, "item": _item()})
    with pytest.raises(ValueError, match="differs"):
        _event(
            adapter,
            {
                "type": "response.completed",
                "response": {"status": "completed", "output": [_item(arguments='{"content":"different"}')]},
            },
        )


@pytest.mark.parametrize("field,value", [("name", "other"), ("type", "message")])
def test_terminal_custom_call_identity_must_match_stream(field, value):
    adapter = NativeResponsesCustomToolAdapter({"freeform": ("freeform", None)})
    _event(adapter, {"type": "response.output_item.added", "output_index": 0, "item": _item(arguments="")})
    _event(adapter, {"type": "response.output_item.done", "output_index": 0, "item": _item()})
    with pytest.raises(ValueError, match="differs from streamed"):
        _event(
            adapter,
            {"type": "response.completed", "response": {"status": "completed", "output": [{**_item(), field: value}]}},
        )
    with pytest.raises(ValueError, match="differs from streamed"):
        _event(adapter, {"type": "response.completed", "response": {"status": "completed", "output": []}})


class _InterruptedStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, error):
        self.error = error

    def __iter__(self):
        yield (
            "data: "
            + json.dumps({"type": "response.output_item.added", "output_index": 0, "item": _item(arguments="")})
            + "\n\n"
        ).encode()
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"content":',
                }
            )
            + "\n\n"
        ).encode()
        raise self.error

    async def __aiter__(self):
        for chunk in self:
            yield chunk


def _interrupted_iterator(error, sync=False):
    config = HostedVLLMResponsesAPIConfig()
    config.transform_responses_api_request(
        "fixture", "hello", {"tools": [{"type": "custom", "name": "freeform"}]}, GenericLiteLLMParams(), {}
    )
    logging = Mock(spec=Logging)
    logging.model_call_details = {"litellm_params": {}}
    logging.completion_start_time = None
    iterator_type = SyncResponsesAPIStreamingIterator if sync else ResponsesAPIStreamingIterator
    return iterator_type(
        httpx.Response(200, stream=_InterruptedStream(error)),
        "fixture",
        config,
        logging,
        custom_llm_provider="hosted_vllm",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ReadError("disconnected"), asyncio.CancelledError()])
async def test_async_interruption_never_completes_buffered_custom_input(error):
    iterator = _interrupted_iterator(error)
    assert (await anext(iterator)).type == "response.output_item.added"
    with pytest.raises(type(error)):
        await anext(iterator)
    assert iterator.completed_response is None
    iterator.logging_obj.success_handler.assert_not_called()


def test_sync_interruption_never_completes_buffered_custom_input():
    iterator = _interrupted_iterator(httpx.ReadError("disconnected"), sync=True)
    assert next(iterator).type == "response.output_item.added"
    with pytest.raises(httpx.ReadError):
        next(iterator)
    assert iterator.completed_response is None
    iterator.logging_obj.success_handler.assert_not_called()


def test_terminal_ids_regenerated_by_backend_use_streamed_custom_identity():
    adapter = NativeResponsesCustomToolAdapter({"freeform": ("freeform", None)})
    item = _item()
    _event(adapter, {"type": "response.output_item.added", "output_index": 0, "item": {**item, "arguments": ""}})
    done = _event(adapter, {"type": "response.output_item.done", "output_index": 0, "item": item})[-1]
    final = _event(
        adapter,
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{**item, "id": "fc_regenerated", "call_id": "call_regenerated"}],
            },
        },
    )[0]
    assert final["response"]["output"][0] == done["item"]
    assert final["response"]["output"][0]["call_id"] == item["call_id"]
    assert final["response"]["output"][0]["id"] == item["id"]
