"""
Test custom_tool_call adaptation for apply_patch and other custom tools.

This test verifies that when Codex sends custom tools (type="custom"),
LiteLLM bridge correctly:
1. Converts them to function tools for Chat Completions providers
2. Converts function_call responses back to custom_tool_call output items
3. Unwraps the JSON-wrapping arguments to extract the actual input content
"""

import json
from typing import Final

import pytest
from openai.types.responses import ResponseFunctionToolCall

from litellm.responses.litellm_completion_transformation.custom_tools import (
    _MAX_ARGUMENTS_LEN,
    build_tool_call_item_kwargs,
    convert_custom_tool_to_function_tool,
    extract_custom_tool_names,
    is_custom_tool_call,
    native_responses_custom_tool_name_map,
    native_responses_namespace_tool_name_map,
    normalize_native_responses_custom_tools,
    openai_shaped_tool_call_item_id,
    unwrap_custom_tool_arguments,
    unwrap_custom_tool_arguments_strict,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.responses.main import CustomToolCallOutputItem


class TestCustomToolUtilities:
    """Test the custom_tools utility functions."""

    def test_extract_custom_tool_names(self):
        """Test extraction of custom tool names from tools list."""
        tools = [
            {"type": "function", "name": "regular_tool"},
            {"type": "custom", "name": "apply_patch"},
            {"type": "function", "name": "another_tool"},
            {"type": "custom", "name": "custom_format"},
        ]

        names = extract_custom_tool_names(tools)
        assert names == {"apply_patch", "custom_format"}

    def test_extract_custom_tool_names_empty(self):
        """Test extraction with no custom tools."""
        tools = [
            {"type": "function", "name": "tool1"},
            {"type": "function", "name": "tool2"},
        ]

        names = extract_custom_tool_names(tools)
        assert names == set()

    def test_extract_custom_tool_names_walks_namespace_tools(self):
        tools = [
            {"type": "function", "name": "regular_tool"},
            {
                "type": "namespace",
                "name": "functions",
                "tools": [
                    {"type": "custom", "name": "exec"},
                    {"type": "function", "name": "wait"},
                    "ignored",
                ],
            },
            {"type": "namespace", "name": "empty", "tools": "not-a-list"},
        ]

        names = extract_custom_tool_names(tools)
        assert names == {"exec"}

    def test_extract_custom_tool_names_none(self):
        """Test extraction with None input."""
        names = extract_custom_tool_names(None)
        assert names == set()

    def test_is_custom_tool_call_true(self):
        """Test identification of custom tool call."""
        custom_names = {"apply_patch", "custom_format"}
        assert is_custom_tool_call("apply_patch", custom_names) is True
        assert is_custom_tool_call("custom_format", custom_names) is True

    def test_is_custom_tool_call_false(self):
        """Test identification of non-custom tool call."""
        custom_names = {"apply_patch"}
        assert is_custom_tool_call("regular_tool", custom_names) is False
        assert is_custom_tool_call("unknown_tool", custom_names) is False

    def test_unwrap_custom_tool_arguments(self):
        """Test unwrapping of JSON-wrapped arguments."""
        # Test with valid JSON
        wrapped = json.dumps({"content": "*** Begin Patch\n*** Add File: test.py\n+hello\n*** End Patch"})
        unwrapped = unwrap_custom_tool_arguments(wrapped)
        assert unwrapped == "*** Begin Patch\n*** Add File: test.py\n+hello\n*** End Patch"

    def test_unwrap_custom_tool_arguments_invalid_json(self):
        """Test unwrapping with invalid JSON returns original."""
        raw = "*** Begin Patch\n*** Add File: test.py\n+hello\n*** End Patch"
        unwrapped = unwrap_custom_tool_arguments(raw)
        assert unwrapped == raw

    def test_unwrap_custom_tool_arguments_no_content_key(self):
        """Test unwrapping with JSON but no content key."""
        wrapped = json.dumps({"other_key": "value"})
        unwrapped = unwrap_custom_tool_arguments(wrapped)
        assert unwrapped == wrapped

    def test_build_tool_call_item_kwargs_custom_completed(self):
        """A completed custom tool call unwraps the content into `input`."""
        wrapped = json.dumps({"content": "patch body"})
        kwargs = build_tool_call_item_kwargs(
            call_id="c1",
            name="apply_patch",
            arguments_or_input=wrapped,
            status="completed",
            custom_tool_names={"apply_patch"},
        )
        assert kwargs["type"] == "custom_tool_call"
        assert kwargs["input"] == "patch body"
        assert "arguments" not in kwargs

    def test_build_tool_call_item_kwargs_custom_in_progress(self):
        """An in-progress custom tool call seeds an empty input string."""
        kwargs = build_tool_call_item_kwargs(
            call_id="c2",
            name="apply_patch",
            arguments_or_input="ignored-until-completed",
            status="in_progress",
            custom_tool_names={"apply_patch"},
        )
        assert kwargs["input"] == ""

    def test_build_tool_call_item_kwargs_regular_function(self):
        """A regular function call keeps raw arguments and uses function_call type."""
        raw = json.dumps({"k": "v"})
        kwargs = build_tool_call_item_kwargs(
            call_id="c3",
            name="get_weather",
            arguments_or_input=raw,
            status="completed",
            custom_tool_names=set(),
        )
        assert kwargs["type"] == "function_call"
        assert kwargs["arguments"] == raw
        assert "input" not in kwargs

    def test_openai_shaped_tool_call_item_id_prefixes_foreign_ids(self):
        """Anthropic-style tool ids must be normalized to OpenAI's item id
        shapes (fc/ctc prefixes) so replaying the item to OpenAI does not 400
        with "Expected an ID that begins with 'fc'"."""
        assert openai_shaped_tool_call_item_id("function_call", "toolu_01Abc") == "fc_toolu_01Abc"
        assert openai_shaped_tool_call_item_id("function_call", "srvtoolu_01Xyz") == "fc_srvtoolu_01Xyz"
        assert openai_shaped_tool_call_item_id("custom_tool_call", "toolu_01Abc") == "ctc_toolu_01Abc"
        assert openai_shaped_tool_call_item_id("function_call", "fc_already") == "fc_already"
        assert openai_shaped_tool_call_item_id("custom_tool_call", "ctc_already") == "ctc_already"
        assert openai_shaped_tool_call_item_id("function_call", "") == ""
        assert openai_shaped_tool_call_item_id("message", "toolu_01Abc") == "toolu_01Abc"

    def test_build_tool_call_item_kwargs_normalizes_item_id_keeps_call_id(self):
        """The streaming item id gets the OpenAI shape while call_id stays raw
        so tool_result pairing (which keys off call_id) keeps working."""
        function_kwargs = build_tool_call_item_kwargs(
            call_id="toolu_01Abc",
            name="get_weather",
            arguments_or_input="{}",
            status="completed",
            custom_tool_names=set(),
        )
        assert function_kwargs["id"] == "fc_toolu_01Abc"
        assert function_kwargs["call_id"] == "toolu_01Abc"

        custom_kwargs = build_tool_call_item_kwargs(
            call_id="toolu_01Def",
            name="apply_patch",
            arguments_or_input=json.dumps({"content": "patch"}),
            status="completed",
            custom_tool_names={"apply_patch"},
        )
        assert custom_kwargs["id"] == "ctc_toolu_01Def"
        assert custom_kwargs["call_id"] == "toolu_01Def"

    def test_unwrap_custom_tool_arguments_oversized_returns_raw(self):
        """Arguments larger than the safety cap are returned unchanged to avoid
        OOM on JSON parsing a pathologically large string."""
        oversized = "x" * (_MAX_ARGUMENTS_LEN + 1)
        assert unwrap_custom_tool_arguments(oversized) == oversized

    def test_unwrap_custom_tool_arguments_empty(self):
        """Empty arguments unwrap to an empty string, not the raw input."""
        assert unwrap_custom_tool_arguments("") == ""

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ("not json", "must contain valid JSON"),
            (json.dumps({"content": {}}), "content must be a string"),
            (json.dumps({"other": "value"}), "must include a content field"),
            (None, "must be a JSON string"),
        ],
    )
    def test_unwrap_custom_tool_arguments_strict_rejects_invalid_envelopes(self, arguments, message):
        with pytest.raises((TypeError, ValueError), match=message):
            unwrap_custom_tool_arguments_strict(arguments)

    def test_unwrap_custom_tool_arguments_strict_returns_only_content(self):
        assert unwrap_custom_tool_arguments_strict(json.dumps({"content": "raw patch"})) == "raw patch"

    @pytest.mark.parametrize("input_value", (None, {"patch": "body"}))
    def test_normalize_native_responses_custom_tools_rejects_non_string_history_input(self, input_value):
        request: Final = {
            "tools": [{"type": "custom", "name": "apply_patch"}],
            "input": [{"type": "custom_tool_call", "name": "apply_patch", "input": input_value}],
        }

        with pytest.raises(ValueError, match="custom tool call input must be a string"):
            normalize_native_responses_custom_tools(request)

    def test_normalize_native_responses_custom_tools_preserves_absent_tool_list(self):
        request: Final = {"tools": None, "input": "continue"}
        assert normalize_native_responses_custom_tools(request) == request

    def test_normalize_native_responses_custom_tools_wraps_request_and_history(self):
        request: Final = {
            "model": "hosted_vllm/model",
            "tools": [
                {"type": "custom", "name": "exec", "description": "Run raw shell input"},
                {
                    "type": "function",
                    "name": "exec",
                    "description": "Run structured shell input",
                    "parameters": {"type": "object"},
                },
            ],
            "input": [
                {
                    "type": "custom_tool_call",
                    "id": "ctc_1",
                    "call_id": "call_1",
                    "name": "exec",
                    "input": "echo hello",
                    "content": "preserved",
                    "status": "completed",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": "hello\n",
                },
            ],
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "custom", "name": "exec"}, {"type": "function", "name": "exec"}],
            },
            "metadata": {"request": "unchanged"},
        }

        custom_tool_names: Final = native_responses_custom_tool_name_map(request)
        wire_name: Final = next(iter(custom_tool_names))
        normalized: Final = normalize_native_responses_custom_tools(request)

        assert custom_tool_names[wire_name] == ("exec", None)
        assert wire_name != "exec"
        assert normalized["metadata"] == request["metadata"]
        assert normalized["tools"] == [
            {
                "type": "function",
                "name": wire_name,
                "description": (
                    f"Call {wire_name} with a JSON object containing the required content string. "
                    "The content field holds the complete tool input. Its description and grammar apply only "
                    "inside that string, not to the outer JSON arguments."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string", "description": "Run raw shell input"}},
                    "required": ["content"],
                },
            },
            request["tools"][1],
        ]
        assert normalized["input"] == [
            {
                "type": "function_call",
                "id": "ctc_1",
                "call_id": "call_1",
                "name": wire_name,
                "content": "preserved",
                "status": "completed",
                "arguments": json.dumps({"content": "echo hello"}),
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "hello\n"},
        ]
        assert normalized["tool_choice"] == {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "function", "name": wire_name}, {"type": "function", "name": "exec"}],
        }

    def test_normalize_native_responses_custom_tools_flattens_nested_custom_tool(self):
        request: Final = {
            "tools": [
                {
                    "type": "namespace",
                    "name": "workspace",
                    "tools": [
                        {"type": "custom", "name": "apply_patch", "description": "Apply a patch"},
                        {"type": "function", "name": "read_file", "parameters": {"type": "object"}},
                    ],
                }
            ],
            "input": [
                {
                    "type": "custom_tool_call",
                    "call_id": "call_patch",
                    "name": "apply_patch",
                    "namespace": "workspace",
                    "input": "*** Begin Patch",
                },
                {
                    "type": "function_call",
                    "call_id": "call_read",
                    "name": "read_file",
                    "namespace": "workspace",
                    "arguments": json.dumps({"path": "README.md"}),
                },
            ],
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [
                    {"type": "custom", "name": "apply_patch", "namespace": "workspace"},
                    {"type": "function", "name": "read_file", "namespace": "workspace"},
                ],
            },
        }

        normalized: Final = normalize_native_responses_custom_tools(request)
        custom_tool_names: Final = native_responses_custom_tool_name_map(request)
        namespace_tool_names: Final = native_responses_namespace_tool_name_map(request)
        wire_name: Final = next(iter(custom_tool_names))

        assert custom_tool_names == {"workspace__apply_patch": ("apply_patch", "workspace")}
        assert namespace_tool_names == {"workspace__read_file": ("workspace", "read_file")}
        assert normalized["tools"][0]["name"] == wire_name
        assert normalized["tools"][1]["name"] == "workspace__read_file"
        assert normalized["input"][0]["name"] == wire_name
        assert normalized["input"][0]["arguments"] == json.dumps({"content": "*** Begin Patch"})
        assert normalized["input"][1] == {
            "type": "function_call",
            "call_id": "call_read",
            "name": "workspace__read_file",
            "arguments": json.dumps({"path": "README.md"}),
        }
        assert normalized["tool_choice"] == {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [
                {"type": "function", "name": wire_name},
                {"type": "function", "name": "workspace__read_file"},
            ],
        }

    def test_normalize_native_responses_custom_tools_flattens_plain_namespace_functions(self):
        request: Final = {
            "tools": [
                {
                    "type": "namespace",
                    "name": "workspace",
                    "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}],
                }
            ],
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_read",
                    "name": "read_file",
                    "namespace": "workspace",
                    "arguments": json.dumps({"path": "README.md"}),
                }
            ],
            "tool_choice": {"type": "function", "name": "read_file", "namespace": "workspace"},
        }

        normalized: Final = normalize_native_responses_custom_tools(request)

        assert normalized["tools"][0]["name"] == "workspace__read_file"
        assert normalized["input"][0]["name"] == "workspace__read_file"
        assert "namespace" not in normalized["input"][0]
        assert normalized["tool_choice"] == {"type": "function", "name": "workspace__read_file"}

    def test_normalize_native_responses_custom_tools_rejects_namespace_name_collision(self):
        request: Final = {
            "tools": [
                {"type": "function", "name": "workspace__read_file", "parameters": {"type": "object"}},
                {
                    "type": "namespace",
                    "name": "workspace",
                    "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}],
                },
            ]
        }

        with pytest.raises(ValueError, match="Top-level function names conflict with flattened namespace tools"):
            normalize_native_responses_custom_tools(request)

    @pytest.mark.parametrize("instructions", [None, "", "Keep the user's existing instructions."])
    def test_native_namespace_description_is_scoped_once_without_changing_tool_contracts(self, instructions):
        shared: Final = "Use these tools only for the temporary workspace. 保留完整说明"
        request: Final = {
            "instructions": instructions,
            "tools": [
                {
                    "type": "namespace",
                    "name": "workspace",
                    "description": shared,
                    "tools": [
                        {"type": "function", "name": "read", "description": "Read a file."},
                        {"type": "function", "name": "list", "description": "List files."},
                        {"type": "custom", "name": "edit", "description": "Apply a patch."},
                    ],
                },
                {"type": "function", "name": "workspace__edit", "description": "Unrelated function."},
                {
                    "type": "namespace",
                    "name": "archive",
                    "description": "Archived files are read-only.",
                    "tools": [{"type": "function", "name": "read", "description": "Read an archived file."}],
                },
            ],
            "input": "Inspect the workspace.",
        }
        before: Final = json.dumps(request)
        normalized: Final = normalize_native_responses_custom_tools(request)
        custom_name: Final = next(iter(native_responses_custom_tool_name_map(request)))

        assert normalized["instructions"] == (
            (instructions + "\n\n" if instructions else "")
            + f'Tool namespace "workspace" ({custom_name}, workspace__read, workspace__list):\n{shared}'
            + '\n\nTool namespace "archive" (archive__read):\nArchived files are read-only.'
        )
        assert json.dumps(normalized, ensure_ascii=False).count(shared) == 1
        assert [tool["description"] for tool in normalized["tools"][1:]] == [
            "Read a file.",
            "List files.",
            "Unrelated function.",
            "Read an archived file.",
        ]
        assert normalized["tools"][-1]["name"] == "archive__read"
        assert normalized["tools"][0]["parameters"]["properties"]["content"]["description"] == "Apply a patch."
        assert normalized["input"] == request["input"]
        assert normalize_native_responses_custom_tools(normalized) == normalized
        assert json.dumps(request) == before

    def test_normalize_native_responses_custom_tools_qualifies_history_without_tools(self):
        request: Final = {
            "input": [
                {
                    "type": "custom_tool_call",
                    "call_id": "call_patch",
                    "name": "apply_patch",
                    "namespace": "workspace",
                    "input": "*** Begin Patch",
                },
                {
                    "type": "function_call",
                    "call_id": "call_read",
                    "name": "read_file",
                    "namespace": "workspace",
                    "arguments": json.dumps({"path": "README.md"}),
                },
            ],
            "tool_choice": {"type": "function", "name": "read_file", "namespace": "workspace"},
        }

        normalized: Final = normalize_native_responses_custom_tools(request)

        assert normalized["input"][0]["name"] == "workspace__apply_patch"
        assert "namespace" not in normalized["input"][0]
        assert normalized["input"][1]["name"] == "workspace__read_file"
        assert "namespace" not in normalized["input"][1]
        assert normalized["tool_choice"] == {"type": "function", "name": "workspace__read_file"}

    def test_convert_custom_tool_to_function_tool_with_format(self):
        raw_description: Final = "Apply a patch. Provide raw patch text, not JSON."
        tool: Final = {
            "type": "custom",
            "name": "apply_patch",
            "description": raw_description,
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": "start: begin_patch",
            },
        }
        result: Final = convert_custom_tool_to_function_tool(tool)
        assert result is not None
        assert result["type"] == "function"
        assert "JSON object" in result["function"]["description"]
        assert raw_description not in result["function"]["description"]
        assert "begin_patch" not in result["function"]["description"]
        assert result["function"]["parameters"]["properties"]["content"] == {
            "type": "string",
            "description": raw_description + "\n\nFormat:\n```lark\nstart: begin_patch\n```",
        }
        assert result["function"]["parameters"]["required"] == ["content"]

    def test_convert_custom_tool_to_function_tool_non_custom_returns_none(self):
        """Non-custom tools are not convertible; the caller keeps them as-is."""
        assert convert_custom_tool_to_function_tool({"type": "function"}) is None


class TestTransformationCustomTools:
    """Test custom tool handling in transformation logic."""

    def test_transform_apply_patch_function_call_to_custom_tool_call(self):
        """Test that apply_patch function_call is converted to custom_tool_call."""
        # Simulate a Chat Completion response with apply_patch function call
        from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse

        tool_call = ChatCompletionMessageToolCall(
            id="call_abc123",
            type="function",
            function=Function(
                name="apply_patch",
                arguments=json.dumps({"content": "*** Begin Patch\n*** Add File: test.py\n+hello\n*** End Patch"}),
            ),
        )

        message = Message(role="assistant", content=None, tool_calls=[tool_call])

        choices = [Choices(index=0, message=message, finish_reason="tool_calls")]

        response = ModelResponse(
            id="test_response", choices=choices, created=1234567890, model="gpt-4", object="chat.completion"
        )

        # Transform with custom tool names
        responses_api_request = {
            "tools": [{"type": "custom", "name": "apply_patch"}, {"type": "function", "name": "regular_tool"}]
        }

        result = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
            response, responses_api_request=responses_api_request
        )

        # Should return a CustomToolCallOutputItem object; ResponsesAPIResponse
        # accepts it directly via its output item union.
        assert len(result) == 1
        item = result[0]
        assert isinstance(item, CustomToolCallOutputItem)
        assert item.type == "custom_tool_call"
        assert item.call_id == "call_abc123"
        assert item.name == "apply_patch"
        assert item.input == "*** Begin Patch\n*** Add File: test.py\n+hello\n*** End Patch"
        assert item.status == "completed"

    def test_custom_tool_call_input_item_recovers_payload_from_input(self):
        """A custom_tool_call input item stores its payload in `input`; the
        assistant tool call must carry it as a JSON content envelope whether
        `arguments` is missing or an empty string."""
        for arguments in (None, ""):
            item = {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "apply_patch",
                "input": "*** Begin Patch\n+hello\n*** End Patch",
            }
            if arguments is not None:
                item["arguments"] = arguments
            messages = (
                LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
                    function_call=item
                )
            )
            tool_call = messages[0]["tool_calls"][0]
            assert tool_call["function"]["arguments"] == json.dumps(
                {"content": "*** Begin Patch\n+hello\n*** End Patch"}
            )

    def test_function_call_input_item_with_empty_arguments_keeps_them_empty(self):
        """A plain function_call input item with empty or missing `arguments`
        must produce an empty arguments string, never a `{"content": ...}`
        envelope (that recovery is reserved for custom_tool_call items) and
        never the literal string "None"."""
        for item in (
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "get_weather",
                "arguments": "",
                "input": "stray value",
            },
            {
                "type": "function_call",
                "call_id": "call_3",
                "name": "get_weather",
            },
        ):
            messages = (
                LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
                    function_call=item
                )
            )
            tool_call = messages[0]["tool_calls"][0]
            assert tool_call["function"]["arguments"] == ""

    def test_transform_regular_function_call_unchanged(self):
        """Test that regular function calls remain as ResponseFunctionToolCall."""
        from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse

        tool_call = ChatCompletionMessageToolCall(
            id="call_xyz789",
            type="function",
            function=Function(name="regular_tool", arguments=json.dumps({"param": "value"})),
        )

        message = Message(role="assistant", content=None, tool_calls=[tool_call])

        choices = [Choices(index=0, message=message, finish_reason="tool_calls")]

        response = ModelResponse(
            id="test_response", choices=choices, created=1234567890, model="gpt-4", object="chat.completion"
        )

        # Transform with custom tool names (regular_tool is NOT custom)
        responses_api_request = {
            "tools": [{"type": "custom", "name": "apply_patch"}, {"type": "function", "name": "regular_tool"}]
        }

        result = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
            response, responses_api_request=responses_api_request
        )

        # Should return ResponseFunctionToolCall
        assert len(result) == 1
        item = result[0]
        assert isinstance(item, ResponseFunctionToolCall)
        assert item.type == "function_call"
        assert item.name == "regular_tool"
        assert item.arguments == json.dumps({"param": "value"})

    def test_transform_anthropic_tool_call_ids_get_openai_item_id_shape(self):
        """Anthropic tool ids (toolu_/srvtoolu_) surfacing through the bridge
        must be emitted with fc/ctc-prefixed item ids so a Responses client can
        replay them to OpenAI verbatim, while call_id stays raw for pairing."""
        from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse

        client_call = ChatCompletionMessageToolCall(
            id="toolu_01ClientCall",
            type="function",
            function=Function(name="get_weather", arguments=json.dumps({"city": "SF"})),
        )
        server_call = ChatCompletionMessageToolCall(
            id="srvtoolu_01ServerCall",
            type="function",
            function=Function(name="web_search", arguments=json.dumps({"query": "zig"})),
        )
        custom_call = ChatCompletionMessageToolCall(
            id="toolu_01CustomCall",
            type="function",
            function=Function(name="apply_patch", arguments=json.dumps({"content": "patch content"})),
        )

        message = Message(role="assistant", content=None, tool_calls=[client_call, server_call, custom_call])
        choices = [Choices(index=0, message=message, finish_reason="tool_calls")]
        response = ModelResponse(
            id="test_response", choices=choices, created=1234567890, model="claude-sonnet-4-5", object="chat.completion"
        )
        responses_api_request = {
            "tools": [{"type": "custom", "name": "apply_patch"}, {"type": "function", "name": "get_weather"}]
        }

        result = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
            response, responses_api_request=responses_api_request
        )

        assert [item.id for item in result] == [
            "fc_toolu_01ClientCall",
            "fc_srvtoolu_01ServerCall",
            "ctc_toolu_01CustomCall",
        ]
        assert [item.call_id for item in result] == [
            "toolu_01ClientCall",
            "srvtoolu_01ServerCall",
            "toolu_01CustomCall",
        ]
        assert result[1].type == "function_call"
        assert result[1].name == "web_search"

    def test_transform_mixed_tool_calls(self):
        """Test transformation with both custom and regular tool calls."""
        from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse

        custom_call = ChatCompletionMessageToolCall(
            id="call_001",
            type="function",
            function=Function(name="apply_patch", arguments=json.dumps({"content": "patch content"})),
        )

        regular_call = ChatCompletionMessageToolCall(
            id="call_002", type="function", function=Function(name="get_weather", arguments=json.dumps({"city": "SF"}))
        )

        message = Message(role="assistant", content=None, tool_calls=[custom_call, regular_call])

        choices = [Choices(index=0, message=message, finish_reason="tool_calls")]

        response = ModelResponse(
            id="test_response", choices=choices, created=1234567890, model="gpt-4", object="chat.completion"
        )

        responses_api_request = {
            "tools": [{"type": "custom", "name": "apply_patch"}, {"type": "function", "name": "get_weather"}]
        }

        result = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
            response, responses_api_request=responses_api_request
        )

        assert len(result) == 2

        # First should be custom_tool_call object
        first = result[0]
        assert isinstance(first, CustomToolCallOutputItem)
        assert first.type == "custom_tool_call"
        assert first.name == "apply_patch"
        assert first.input == "patch content"

        # Second should be function_call
        second = result[1]
        assert isinstance(second, ResponseFunctionToolCall)
        assert second.type == "function_call"
        assert second.name == "get_weather"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
