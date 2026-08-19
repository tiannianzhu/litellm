# Hosted vLLM compatibility notes

This file tracks local LiteLLM adaptations that can be removed when vLLM or
upstream LiteLLM provides the same behavior. Verify each removal condition
against the deployed vLLM version before deleting its code and tests.

## Reasoning effort

Model deployments use either native `reasoning_effort` values or
`chat_template_kwargs`. LiteLLM normalizes OpenAI and Anthropic reasoning
controls according to each deployment's `model_info.reasoning_effort` config.

- Code: `litellm/llms/hosted_vllm/reasoning.py` plus the hosted-vLLM chat,
  Messages, and Responses transformations; `reasoning_effort_config` forwarding
  in `litellm/main.py`, `litellm/utils.py`, and `litellm/constants.py`
- Remove when: all deployed vLLM models accept the client-facing reasoning
  controls directly with the same disable and effort-level semantics
- Tests: the chat, Messages, and Responses tests below
  `tests/test_litellm/llms/hosted_vllm/`

## Cache creation usage

Some vLLM Chat Completions payloads expose cache creation usage as
`created_cache_tokens`. LiteLLM maps it to `cache_write_tokens`, which feeds
`cache_creation_input_tokens` and cache-write cost accounting.

- Code: `litellm/types/utils.py`, `PromptTokensDetailsWrapper`
- Remove when: upstream LiteLLM performs this mapping, or vLLM consistently
  emits the standard cache-creation field consumed by LiteLLM
- Tests: `tests/test_litellm/test_utils.py`

## Responses custom tools

vLLM's native Responses endpoint parses historical `custom_tool_call` items
but then treats them as dictionaries, causing
`'ResponseCustomToolCall' object has no attribute 'get'`. LiteLLM routes
hosted-vLLM requests containing custom tool definitions or history through
the existing Chat Completions bridge and reconstructs Responses streaming
events for clients.

- Observed on: vLLM `0.27.2rc1.dev122+g8efa13b70`; still present in vLLM
  `0.28.0`
- Code: `_hosted_vllm_request_requires_chat_completions()` in
  `litellm/responses/main.py`
- Supporting generic bridge code:
  `litellm/responses/litellm_completion_transformation/streaming_iterator.py`,
  `litellm/responses/streaming_iterator.py`, and `litellm/proxy/utils.py`
- Remove when: the deployed vLLM accepts paired `custom_tool_call` and
  `custom_tool_call_output` history on `/v1/responses`, including streaming,
  without this bridge
- Removal scope: remove the hosted-vLLM routing helper and its call first. Keep
  the generic bridge code while other bridged providers or clients depend on it
- Tests: `tests/test_litellm/responses/test_custom_tool_call.py`,
  `tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py`,
  and
  `tests/test_litellm/proxy/test_common_request_processing.py`
