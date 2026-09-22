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

## Tool schemas

Chat Completions forwards function-level `strict` and JSON Schema constraints,
including nested `additionalProperties: false`, unchanged. Custom-to-function
conversion also preserves the supplied input schema. Constraint enforcement
depends on the deployed vLLM version, tool parser, and tool choice; forwarding
these fields does not guarantee enforcement for every configuration


## Explicit Responses to Chat bridge

Set `use_chat_completions_api: true` in a hosted-vLLM deployment's `litellm_params`
to accept client Responses requests while sending `/v1/chat/completions` to that
backend. The flag is opt-in; native Responses remains the default. Start with
Codex `use_responses_lite=false` and local conversation history replay

The bridge preserves plaintext reasoning as `reasoning_text` content in both
non-streaming output and streaming `output_item.done` / terminal snapshots.
Replayed content parts retain their text and order, and hosted Chat retains
`reasoning_content`. Adjacent reasoning items merge into Chat assistant messages;
original Responses item boundaries cannot always be reconstructed. Tool call IDs
remain the correlation key across continuation requests

Hosted bridge requests use the shared custom and namespace name mapping,
including collision checks, history, named choices, and one scoped copy of each
namespace description. `allowed_tools` filters the outgoing function definitions
and maps its mode to Chat `auto` or `required`. Custom envelopes must contain
valid string content before any executable input is returned. Arbitrary custom
grammars remain descriptive, not enforced by the function schema

Request tools, instructions, metadata, reasoning options, and the parallel flag
are reflected in returned Responses metadata. This does not make Chat enforce
unsupported backend capabilities. Usage comes from the backend. Length and
content-filter endings produce incomplete responses; transport exceptions
propagate as errors

Text tool-result parts retain their order. Images, files, and unknown parts in
tool results are rejected locally for hosted vLLM instead of being silently
omitted: its Chat tool-message schema supports text. Opaque encrypted reasoning
is also rejected locally. Signed reasoning for other providers keeps its
existing conversion path

This scope does not establish Lite, WebSocket, hosted search, nondefault
`reasoning.context`, item-reference lookup, or server-side `previous_response_id`
parity. It does not decrypt official encrypted reasoning or promise identical
native/Chat token counts. Removing the flag sends requests to vLLM's native
Responses endpoint without the Chat bridge's custom and namespace adaptations
