---
name: inspect-errors
description: Inspect this LiteLLM deployment's SpendLogs and bounded runtime logs to classify failures, separate client and policy errors from gateway or model-service faults, attribute load by end-user alias, and assess timeout or queue pressure. Use for recurring error checks and operational incident diagnosis; do not use for code-only debugging without deployment evidence.
---

# Inspect Errors

Use SpendLogs as the primary evidence source, then use bounded container logs only to explain a specific unresolved service-side pattern. Keep the investigation read-only unless the user separately asks for a configuration or code change

## Read the evidence

Run the bundled report from the repository root:

```bash
.agents/skills/inspect-errors/scripts/spendlogs_report.sh
```

The analysis range is the earliest through latest `endTime` among every currently retained `status='failure'` row. `scripts/cleanup_usage_noise.sh` removes those rows after review, so this range naturally starts over after cleanup. Never use the full success-log history or a fixed recent-time window as the analysis range

The report classifies every retained failure, caller group, and normalized error signature. Count successes only inside the failure-row range when comparing health. A success after the latest failure is recovery evidence, but it does not extend the analysis range. The report is read-only and defaults to container `litellm-db-1`, user `litellm`, and database `litellm`. Override these with `SPENDLOGS_DB_CONTAINER`, `SPENDLOGS_DB_USER`, and `SPENDLOGS_DB_NAME` when deployment discovery shows different names. Do not print or inspect database passwords just to run an in-container local connection

## Classify before diagnosing

Keep these classes separate in the final answer:

- `expected_policy`: intentional gateway rules such as rejecting attempts to disable always-on reasoning
- `client_*`: malformed messages, tools or input, context and max-token overflow, unsupported images, invalid model names, and expired credentials
- `backend_*`: socket read timeouts, connection timeouts, unavailable engines, connection resets, and unexplained 5xx responses
- `capacity_or_rate_limit`: queue, concurrency, RPM, TPM, ITPM, OTPM, or upstream overload signals
- `unclassified`: patterns whose stored error text is insufficient for attribution

Do not call every HTTP 500 a service fault. LiteLLM often wraps upstream validation and client format failures. Use the innermost error message, class, model, timing, and cross-user distribution

Do not count retries or companion records as independent incidents without checking `litellm_call_id`, `attempted_retries`, timestamps, and message similarity. Report both failure rows and distinct incidents when the difference is material

## Decide what needs attention

Treat a pattern as a likely service issue when it is distributed across unrelated callers, clusters in time, uses valid requests, and is followed by socket, connection, engine, overload, or 5xx failures. Check whether successes resume after the last failure before calling an outage current

For timeouts and queue pressure, compare prompt-token distributions and caller concentration. RPM alone treats a short request and a 250k-token request equally. Inspect ITPM, max parallel requests, active retry settings, client retry cadence, and time-to-first-byte evidence before blaming request volume alone

Use `LiteLLM_EndUserTable.alias` and the API key alias in SpendLogs for attribution. Prefer names and aliases over raw UUIDs. If an alias is absent, show the UUID without guessing an identity

Known intentional policy errors remain visible in counts but must not be presented as service failures. A repeated malformed request from one caller is a client integration problem even when LiteLLM returns a wrapped 500

## Escalate to runtime logs narrowly

Only inspect runtime logs after identifying a model, error signature, and time interval from SpendLogs. Bound `docker logs` with `--since` and filter for the request ID, model, upstream host, or exact exception. Do not dump whole logs, prompts, message bodies, API keys, or environment values into the conversation

When the question is about queue accumulation, inspect the serving backend's queue and scheduler metrics if available, then correlate them with SpendLogs by time. Separate gateway waiting, upstream waiting, active generation, and client-side retries

## Report the outcome

Lead with whether a service problem is still active and state the Beijing-time failure-row range and total failure-row count

Always follow that opening with a Markdown table in this shape:

| 分类 | 数量 | 时间范围 | 判断 |
| --- | ---: | --- | --- |

Include one row for every observed category. Also include zero-count rows for `capacity_or_rate_limit` and combined `backend_5xx` or `unclassified` when absent, because their absence is useful evidence. Use the category's first and last retained failure timestamps, not a fixed recent window. Write a concise operational judgment in the final column. Do not replace this table with bullets or raw script output

After the table, summarize affected models, attributed callers, token-length evidence, runtime-log findings, and recovery after the last failure when they materially change the diagnosis. Base every failure count and classification on the retained failure rows. Clearly label intentional rules and client errors so they do not obscure actual backend faults

Do not change configuration, restart containers, clear queues, disable users, or send work to another task unless the user explicitly requests that action
