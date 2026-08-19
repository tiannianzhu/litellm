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

- Code: `litellm/llms/hosted_vllm/reasoning.py` plus the hosted-vLLM chat and
  Messages transformations; `reasoning_effort_config` forwarding
  in `litellm/main.py`, `litellm/utils.py`, and `litellm/constants.py`
- Remove when: all deployed vLLM models accept the client-facing reasoning
  controls directly with the same disable and effort-level semantics
- Tests: the chat and Messages tests below
  `tests/test_litellm/llms/hosted_vllm/`

## Cache creation usage

Some vLLM Chat Completions payloads expose cache creation usage as
`created_cache_tokens`. LiteLLM maps it to `cache_write_tokens`, which feeds
`cache_creation_input_tokens` and cache-write cost accounting.

- Code: `litellm/types/utils.py`, `PromptTokensDetailsWrapper`
- Remove when: upstream LiteLLM performs this mapping, or vLLM consistently
  emits the standard cache-creation field consumed by LiteLLM
- Tests: `tests/test_litellm/test_utils.py`
