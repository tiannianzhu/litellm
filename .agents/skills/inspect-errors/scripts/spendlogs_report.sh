#!/usr/bin/env bash
set -euo pipefail

db_container="${SPENDLOGS_DB_CONTAINER:-litellm-db-1}"
db_user="${SPENDLOGS_DB_USER:-litellm}"
db_name="${SPENDLOGS_DB_NAME:-litellm}"

docker exec -i "$db_container" psql -X -v ON_ERROR_STOP=1 -U "$db_user" -d "$db_name" \
  -P pager=off -F $'\t' -A <<'SQL'
\pset footer off
\pset null '<none>'

\echo 'ERROR_LOG_RANGE'
SELECT NOW() AT TIME ZONE 'Asia/Shanghai' AS generated_at_beijing,
       COUNT(*) AS failure_rows,
       TO_CHAR(MIN("endTime") AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai',
               'YYYY-MM-DD HH24:MI:SS') AS first_failure_beijing,
       TO_CHAR(MAX("endTime") AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai',
               'YYYY-MM-DD HH24:MI:SS') AS last_failure_beijing
FROM "LiteLLM_SpendLogs"
WHERE status = 'failure';

\echo 'FAILURE_CATEGORY_SUMMARY'
WITH category_order(category, sort_order) AS (
  VALUES
    ('backend_timeout', 1),
    ('backend_unavailable', 2),
    ('backend_connection_closed', 3),
    ('capacity_or_rate_limit', 4),
    ('backend_5xx', 5),
    ('client_context_limit', 6),
    ('client_request_format', 7),
    ('client_model_mismatch', 8),
    ('client_invalid_model', 9),
    ('client_auth_or_key', 10),
    ('expected_policy', 11),
    ('unclassified', 12)
), failures AS (
  SELECT COALESCE(NULLIF(model_group, ''), NULLIF(model, ''), '<unknown>') AS model_name,
         COALESCE(metadata->'error_information'->>'error_message', '') AS message,
         COALESCE(metadata->'error_information'->>'error_class', '') AS error_class,
         COALESCE(metadata->'error_information'->>'error_code', '') AS error_code,
         "endTime"
  FROM "LiteLLM_SpendLogs"
  WHERE status = 'failure'
), classified AS (
  SELECT *,
         CASE
           WHEN message ILIKE '%always has reasoning enabled and cannot be disabled%' THEN 'expected_policy'
           WHEN message ILIKE '%maximum context length%'
             OR message ILIKE '%max_tokens=%greater than%'
             OR error_class = 'ContextWindowExceededError' THEN 'client_context_limit'
           WHEN message ILIKE '%validation errors%'
             OR message ILIKE '%body.tools%'
             OR message ILIKE '%missing field%type%'
             OR message ILIKE '%input should be function%'
             OR message ILIKE '%system message%beginning%'
             OR message ILIKE '%system message%start%'
             OR message ILIKE '%tool_choice%incompatible%' THEN 'client_request_format'
           WHEN message ILIKE '%not a multimodal model%' THEN 'client_model_mismatch'
           WHEN message ILIKE '%invalid model name%' OR error_class = 'NotFoundError' THEN 'client_invalid_model'
           WHEN message ILIKE '%expired key%'
             OR message ILIKE '%invalid proxy server token%'
             OR error_code IN ('401', '403')
             OR error_class IN ('AuthenticationError', 'PermissionDeniedError', 'KeyNotFoundError')
             THEN 'client_auth_or_key'
           WHEN message ILIKE '%timeout on reading data from socket%'
             OR message ILIKE '%connection timed out%'
             OR message ILIKE '%timeout passed=%'
             OR error_class = 'Timeout' THEN 'backend_timeout'
           WHEN message ILIKE '%enginecore%'
             OR message ILIKE '%connection refused%'
             OR message ILIKE '%failed to connect%'
             OR message ILIKE '%no healthy deployment%'
             OR message ILIKE '%service unavailable%' THEN 'backend_unavailable'
           WHEN message ILIKE '%connection closed%'
             OR message ILIKE '%connection reset%'
             OR message ILIKE '%server disconnected%' THEN 'backend_connection_closed'
           WHEN error_code = '429'
             OR error_class = 'RateLimitError'
             OR message ILIKE '%rate limit%'
             OR message ILIKE '%overload%'
             OR message ILIKE '%queue%full%' THEN 'capacity_or_rate_limit'
           WHEN error_code ~ '^5[0-9][0-9]$' THEN 'backend_5xx'
           ELSE 'unclassified'
         END AS category
  FROM failures
)
SELECT o.category,
       COUNT(c.category) AS failure_rows,
       TO_CHAR(MIN(c."endTime") AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai',
               'YYYY-MM-DD HH24:MI:SS') AS first_beijing,
       TO_CHAR(MAX(c."endTime") AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai',
               'YYYY-MM-DD HH24:MI:SS') AS last_beijing,
       COALESCE(STRING_AGG(DISTINCT c.model_name, ', ' ORDER BY c.model_name), '<none>') AS affected_models
FROM category_order o
LEFT JOIN classified c ON c.category = o.category
GROUP BY o.category, o.sort_order
ORDER BY o.sort_order;

\echo 'MODEL_HEALTH'
WITH error_bounds AS (
  SELECT MIN("endTime") AS first_failure,
         MAX("endTime") AS last_failure
  FROM "LiteLLM_SpendLogs"
  WHERE status = 'failure'
), scoped_requests AS (
  SELECT COALESCE(NULLIF(model_group, ''), NULLIF(model, ''), '<unknown>') AS model_name,
         status,
         "endTime"
  FROM "LiteLLM_SpendLogs"
  CROSS JOIN error_bounds
  WHERE first_failure IS NOT NULL
    AND "endTime" BETWEEN first_failure AND last_failure
), scoped_health AS (
  SELECT model_name,
         COUNT(*) FILTER (WHERE status = 'success') AS successes_in_error_range,
         COUNT(*) FILTER (WHERE status = 'failure') AS failures_in_error_range,
         MAX("endTime") FILTER (WHERE status = 'failure') AS last_failure
  FROM scoped_requests
  GROUP BY model_name
  HAVING COUNT(*) FILTER (WHERE status = 'failure') > 0
), latest_success AS (
  SELECT COALESCE(NULLIF(model_group, ''), NULLIF(model, ''), '<unknown>') AS model_name,
         MAX("endTime") AS last_success
  FROM "LiteLLM_SpendLogs"
  WHERE status = 'success'
  GROUP BY COALESCE(NULLIF(model_group, ''), NULLIF(model, ''), '<unknown>')
)
SELECT h.model_name,
       h.successes_in_error_range,
       h.failures_in_error_range,
       ROUND(100.0 * h.failures_in_error_range
             / NULLIF(h.successes_in_error_range + h.failures_in_error_range, 0), 2) AS failure_percent_in_range,
       TO_CHAR(h.last_failure AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai',
               'YYYY-MM-DD HH24:MI:SS') AS last_failure_beijing,
       TO_CHAR(s.last_success AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai',
               'YYYY-MM-DD HH24:MI:SS') AS latest_success_beijing,
       CASE
         WHEN s.last_success > h.last_failure THEN 'yes'
         WHEN s.last_success IS NULL THEN 'no_success_recorded'
         ELSE 'no'
       END AS recovered_after_last_failure
FROM scoped_health h
LEFT JOIN latest_success s USING (model_name)
ORDER BY h.failures_in_error_range DESC;

\echo 'FAILURE_CALLERS'
WITH failures AS (
  SELECT COALESCE(NULLIF(s.model_group, ''), NULLIF(s.model, ''), '<unknown>') AS model_name,
         COALESCE(NULLIF(s.end_user, ''), NULLIF(s.metadata->>'user_api_key_user_id', ''), '<unknown>') AS enduser_id,
         s.metadata->>'user_api_key_alias' AS key_alias,
         COALESCE(s.metadata->'error_information'->>'error_message', '') AS message,
         COALESCE(s.metadata->'error_information'->>'error_class', '') AS error_class,
         COALESCE(s.metadata->'error_information'->>'error_code', '') AS error_code,
         s.prompt_tokens,
         s."endTime"
  FROM "LiteLLM_SpendLogs" s
  WHERE s.status = 'failure'
), classified AS (
  SELECT *,
         CASE
           WHEN message ILIKE '%always has reasoning enabled and cannot be disabled%' THEN 'expected_policy'
           WHEN message ILIKE '%maximum context length%'
             OR message ILIKE '%max_tokens=%greater than%'
             OR error_class = 'ContextWindowExceededError' THEN 'client_context_limit'
           WHEN message ILIKE '%validation errors%'
             OR message ILIKE '%body.tools%'
             OR message ILIKE '%missing field%type%'
             OR message ILIKE '%input should be function%'
             OR message ILIKE '%system message%beginning%'
             OR message ILIKE '%system message%start%'
             OR message ILIKE '%tool_choice%incompatible%' THEN 'client_request_format'
           WHEN message ILIKE '%not a multimodal model%' THEN 'client_model_mismatch'
           WHEN message ILIKE '%invalid model name%' OR error_class = 'NotFoundError' THEN 'client_invalid_model'
           WHEN message ILIKE '%expired key%'
             OR message ILIKE '%invalid proxy server token%'
             OR error_code IN ('401', '403')
             OR error_class IN ('AuthenticationError', 'PermissionDeniedError', 'KeyNotFoundError')
             THEN 'client_auth_or_key'
           WHEN message ILIKE '%timeout on reading data from socket%'
             OR message ILIKE '%connection timed out%'
             OR message ILIKE '%timeout passed=%'
             OR error_class = 'Timeout' THEN 'backend_timeout'
           WHEN message ILIKE '%enginecore%'
             OR message ILIKE '%connection refused%'
             OR message ILIKE '%failed to connect%'
             OR message ILIKE '%no healthy deployment%'
             OR message ILIKE '%service unavailable%' THEN 'backend_unavailable'
           WHEN message ILIKE '%connection closed%'
             OR message ILIKE '%connection reset%'
             OR message ILIKE '%server disconnected%' THEN 'backend_connection_closed'
           WHEN error_code = '429'
             OR error_class = 'RateLimitError'
             OR message ILIKE '%rate limit%'
             OR message ILIKE '%overload%'
             OR message ILIKE '%queue%full%' THEN 'capacity_or_rate_limit'
           WHEN error_code ~ '^5[0-9][0-9]$' THEN 'backend_5xx'
           ELSE 'unclassified'
         END AS category
  FROM failures
)
SELECT c.category,
       c.model_name,
       COALESCE(NULLIF(e.alias, ''), NULLIF(c.enduser_id, ''), '<unknown>') AS enduser,
       COALESCE(c.key_alias, '<none>') AS key_alias,
       COUNT(*) AS failure_rows,
       ROUND(AVG(c.prompt_tokens)) AS avg_prompt_tokens,
       MAX(c.prompt_tokens) AS max_prompt_tokens,
       TO_CHAR(MAX(c."endTime" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai'),
               'YYYY-MM-DD HH24:MI:SS') AS last_beijing
FROM classified c
LEFT JOIN "LiteLLM_EndUserTable" e ON e."user_id" = c.enduser_id
GROUP BY c.category, c.model_name,
         COALESCE(NULLIF(e.alias, ''), NULLIF(c.enduser_id, ''), '<unknown>'),
         COALESCE(c.key_alias, '<none>')
ORDER BY failure_rows DESC, last_beijing DESC;

\echo 'FAILURE_SIGNATURES'
WITH signatures AS (
  SELECT COALESCE(NULLIF(model_group, ''), NULLIF(model, ''), '<unknown>') AS model_name,
         COALESCE(metadata->'error_information'->>'error_class', '<none>') AS error_class,
         COALESCE(metadata->'error_information'->>'error_code', '<none>') AS error_code,
         CASE
           WHEN COALESCE(metadata->'error_information'->>'error_code', '') IN ('401', '403')
             OR COALESCE(metadata->'error_information'->>'error_class', '')
                IN ('AuthenticationError', 'PermissionDeniedError', 'KeyNotFoundError')
             THEN '<authentication error redacted>'
           ELSE LEFT(
             REGEXP_REPLACE(COALESCE(metadata->'error_information'->>'error_message', '<blank>'),
                            E'[\\n\\r\\t]+', ' ', 'g'),
             240
           )
         END AS error_signature,
         "endTime"
  FROM "LiteLLM_SpendLogs"
  WHERE status = 'failure'
)
SELECT model_name,
       error_class,
       error_code,
       error_signature,
       COUNT(*) AS failure_rows,
       TO_CHAR(MIN("endTime" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai'),
               'YYYY-MM-DD HH24:MI:SS') AS first_beijing,
       TO_CHAR(MAX("endTime" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai'),
               'YYYY-MM-DD HH24:MI:SS') AS last_beijing
FROM signatures
GROUP BY model_name, error_class, error_code, error_signature
ORDER BY failure_rows DESC, last_beijing DESC;
SQL
