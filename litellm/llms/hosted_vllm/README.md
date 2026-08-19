# Hosted vLLM compatibility notes

This file tracks local LiteLLM adaptations that can be removed when vLLM or
upstream LiteLLM provides the same behavior. Verify each removal condition
against the deployed vLLM version before deleting its code and tests.

## Reasoning effort

LiteLLM normalizes standard OpenAI and Anthropic reasoning controls to native
vLLM effort fields, with a shared default of `high`. Deployment configuration
only selects effort aliases and whether disabling is rejected. Messages thinking
switches are translated internally to `enable_thinking`, because vLLM ignores
the standard `thinking` field

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

Hosted vLLM keeps requests on `/v1/responses`. LiteLLM wraps custom tools as
functions with one required `content` string and translates paired custom-call
history to function-call history, preserving call IDs. Namespaced functions use
the existing qualified-name mapping. Return conversion uses the tools wrapped
for that request, so ordinary function calls retain their type

The original custom format and grammar remain in the content parameter's
description. This transports the input; it does not enforce arbitrary grammars
through the function JSON schema. Custom argument fragments are buffered until
complete, strictly decoded, and emitted as custom input delta/done events.
Malformed envelopes fail instead of being returned as executable input. Other
native events continue streaming, including reasoning, text, and usage. If vLLM
regenerates custom call IDs in the completed snapshot, LiteLLM retains the
streamed IDs after checking the output position, tool name, and decoded input

- Request helpers: `litellm/responses/litellm_completion_transformation/custom_tools.py`
- Native provider and return adapter: `litellm/llms/hosted_vllm/responses/`
- Tests: `tests/test_litellm/llms/hosted_vllm/responses/` and
  `tests/test_litellm/responses/test_custom_tool_call.py`

The generic Chat Completions bridge remains available to other providers and
explicit callers. Custom tools no longer force hosted vLLM onto that bridge.
Web search interception still uses its existing non-streaming agentic loop and
synthetic Responses events; this adapter does not change search execution

## Responses namespace functions

Keep the ordinary namespace function mapping until deployed backends support
explicit named tool choices as well as automatic selection. Native automatic
selection can preserve namespaced calls and replay their history while a named
choice still fails with a tool-not-found error, even when its name is qualified
as `namespace__function`. Flattening both the definitions and the choice avoids
that failure. Verify same-named functions in different namespaces, streaming
output, and history replay before removing the mapping

## Responses developer messages

For a backend whose chat encoder rejects the `developer` role, set
`model_info.supports_developer_messages: false` on that deployment. Native
Responses then maps developer messages to system messages, retaining message
order, content parts, other message fields, and the original system instructions.
Omitting the setting or setting it to `true` preserves developer messages

This is a compatibility mapping for the backend's limited role vocabulary. It
cannot retain a distinction between system and developer priority that the
backend itself does not represent. It applies to ordinary conversations and
structured output alike, independently of tools or prompt text

- Code and tests: the hosted-vLLM Responses transformation and its mapped tests
- Remove or disable per deployment when its encoder accepts developer messages
