import json
from base64 import b64decode
from collections.abc import Callable
from hashlib import sha256
from typing import Final
from uuid import uuid4

import pytest
from integration._support.client import Gateway, eventually, list_value, object_value
from integration._support.database import read_rows
from integration._support.responses_stream import data_frames
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

PROMPT_TOKENS: Final = 30
COMPLETION_TOKENS: Final = 5
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
EXPECTED_SPEND: Final = PROMPT_TOKENS * INPUT_RATE + COMPLETION_TOKENS * OUTPUT_RATE


def _responses_reply(prompt: str, response_id: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(
                body=json.dumps({"object": "list", "data": [{"id": "gpt-6.1-sol", "object": "model"}]}).encode()
            )
        assert request.method == "POST", request.method
        assert request.target == "/v1/responses", request.target
        body: Final = object_value(json.loads(request.body))
        encoded_input: Final = json.dumps(body["input"])
        assert prompt in encoded_input, encoded_input
        assert body.get("stream") is not True, body
        response: Final = {
            "id": response_id,
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": body["model"],
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{uuid4().hex}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": PROMPT_TOKENS,
                "output_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            },
        }
        return Reply(body=json.dumps(response).encode())

    return respond


def _rows_for_key(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT litellm_call_id, request_id, spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
        (sha256(key.encode()).hexdigest(),),
    )


@pytest.mark.timeout(120)
def test_bridged_chat_completion_above_log_offload_threshold_logs_and_charges_once(gateway: Gateway) -> None:
    prompt_prefix: Final = f"spend once {uuid4().hex[:8]} "
    long_prompt: Final = (prompt_prefix + "lorem ipsum " * 30_000)[:300_000]
    upstream_id: Final = f"resp_{uuid4().hex}"
    with (
        wire_server(_responses_reply(long_prompt, upstream_id)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="openai/responses/gpt-6.1-sol",
            api_base=f"{wire.url}/v1",
            input_cost_per_token=INPUT_RATE,
            output_cost_per_token=OUTPUT_RATE,
        )
        key: Final = scenario.key(models=[model])
        digest: Final = sha256(key.encode()).hexdigest()

        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": long_prompt}], "stream": False},
            key=key,
        )
        assert response.status_code == 200, response.text
        assert "ok" in response.text, response.text
        assert response.json()["id"] == upstream_id, response.text
        call_id: Final = response.headers["x-litellm-call-id"]

        rows: Final = eventually(
            lambda: _rows_for_key(key),
            lambda values: any(row["litellm_call_id"] == call_id for row in values),
            seconds=70,
        )
        call_rows: Final = [row for row in rows if row["litellm_call_id"] == call_id]
        assert len(call_rows) == 1, rows
        assert float(str(call_rows[0]["spend"])) == pytest.approx(EXPECTED_SPEND), rows
        logged_id: Final = b64decode(str(call_rows[0]["request_id"]).removeprefix("resp_")).decode()
        assert logged_id.startswith("litellm:") and f";response_id:{upstream_id}" in logged_id, rows

        key_spend: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
            lambda values: len(values) == 1 and float(str(values[0]["spend"])) >= EXPECTED_SPEND,
            seconds=70,
        )
        assert float(str(key_spend[0]["spend"])) == pytest.approx(EXPECTED_SPEND), key_spend


@pytest.mark.parametrize("valid_tool_input", [False, True], ids=["rejected-tool-input", "valid-tool-input"])
def test_responses_chat_bridge_stream_records_final_status_and_charges_once(
    gateway: Gateway, valid_tool_input: bool
) -> None:
    marker: Final = uuid4().hex
    arguments: Final = json.dumps({"content" if valid_tool_input else "input": "synthetic tool input"})
    chunk: Final = {
        "id": f"chatcmpl-{marker}",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "test-model",
    }
    frames: Final = (
        {
            **chunk,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": f"call_{marker}",
                                "type": "function",
                                "function": {"name": "edit", "arguments": arguments},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {
            **chunk,
            "choices": [],
            "usage": {
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            },
        },
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        body: Final = object_value(json.loads(request.body))
        assert body["stream"] is True, body
        tool: Final = object_value(object_value(list_value(body["tools"])[0])["function"])
        schema: Final = object_value(tool["parameters"])
        assert schema["required"] == ["content"], schema
        assert object_value(object_value(schema["properties"])["content"])["type"] == "string", schema
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(f"data: {json.dumps(frame)}\n\n".encode() for frame in frames) + (b"data: [DONE]\n\n",),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="hosted_vllm/test-model",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-provider-key",
            use_chat_completions_api=True,
            input_cost_per_token=INPUT_RATE,
            output_cost_per_token=OUTPUT_RATE,
            num_retries=0,
        )
        key: Final = scenario.key(models=[model])
        digest: Final = sha256(key.encode()).hexdigest()
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": marker, "tools": [{"type": "custom", "name": "edit"}], "stream": True},
            key=key,
        )
        assert response.status_code == 200, response.text
        events: Final = data_frames(response.text)
        assert sum(event.get("type") == "response.completed" for event in events) == int(valid_tool_input), (
            response.text
        )
        if valid_tool_input:
            assert any(event.get("input") == "synthetic tool input" for event in events), response.text
        else:
            assert "custom tool arguments must include a content field" in response.text, response.text
        assert len(wire.drain()) == 1

        rows: Final = eventually(
            lambda: read_rows(
                'SELECT status, prompt_tokens, completion_tokens, total_tokens, spend FROM "LiteLLM_SpendLogs" '
                "WHERE api_key=%s",
                (digest,),
            ),
            bool,
            seconds=70,
        )
        assert rows == [
            {
                "status": "success" if valid_tool_input else "failure",
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                "spend": pytest.approx(EXPECTED_SPEND),
            }
        ], rows
        key_spend: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
            lambda values: len(values) == 1 and float(str(values[0]["spend"])) >= EXPECTED_SPEND,
            seconds=70,
        )
        assert float(str(key_spend[0]["spend"])) == pytest.approx(EXPECTED_SPEND), key_spend
        daily: Final = eventually(
            lambda: read_rows(
                "SELECT api_requests, successful_requests, failed_requests, prompt_tokens, completion_tokens, spend "
                'FROM "LiteLLM_DailyUserSpend" WHERE api_key=%s',
                (digest,),
            ),
            bool,
            seconds=70,
        )
        assert daily == [
            {
                "api_requests": 1,
                "successful_requests": int(valid_tool_input),
                "failed_requests": int(not valid_tool_input),
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "spend": pytest.approx(EXPECTED_SPEND),
            }
        ], daily
