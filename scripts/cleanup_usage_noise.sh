#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
	cat <<'EOF'
Usage: scripts/cleanup_usage_noise.sh [--dry-run|--apply]

Removes failed requests and internal health/master-key traffic from SpendLogs,
their cold-storage objects, and the corresponding daily usage and gateway aggregates.

  --dry-run   Show how many rows would be changed. This is the default.
  --apply     Apply the cleanup transaction.
EOF
}

if (($# > 1)); then
	usage >&2
	exit 2
fi

ACTION="${1:---dry-run}"
case "$ACTION" in
--dry-run | --apply) ;;
-h | --help)
	usage
	exit 0
	;;
*)
	usage >&2
	exit 2
	;;
esac

if [[ -f .env.production ]]; then
	set -a
	# shellcheck disable=SC1091
	. ./.env.production
	set +a
fi

if [[ -z "${POSTGRES_USER:-}" || -z "${POSTGRES_DB:-}" ]]; then
	echo "POSTGRES_USER and POSTGRES_DB must be set in .env.production" >&2
	exit 1
fi

COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-litellm}"
COMPOSE_FILES=(-f docker-compose.yml)
if [[ -f docker-compose.prod.yml ]]; then
	COMPOSE_FILES+=(-f docker-compose.prod.yml)
fi

COMPOSE=(
	docker compose
	-p "$COMPOSE_PROJECT_NAME"
	"${COMPOSE_FILES[@]}"
)

ROUTE_ENDPOINT_MAPPING=$("${COMPOSE[@]}" exec -T litellm python -c \
	'import json; from litellm.proxy.route_llm_request import ROUTE_ENDPOINT_MAPPING; print(json.dumps(ROUTE_ENDPOINT_MAPPING, separators=(",", ":")))')

PSQL=(
	"${COMPOSE[@]}"
	exec -T db
	psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"
	-q -v ON_ERROR_STOP=1 -v "route_endpoint_mapping=$ROUTE_ENDPOINT_MAPPING" -P pager=off
)

DAILY_TABLES=(
	LiteLLM_DailyUserSpend
	LiteLLM_DailyTeamSpend
	LiteLLM_DailyTagSpend
	LiteLLM_DailyEndUserSpend
)

internal_noise_values_sql() {
	cat <<'SQL'
('litellm-internal-health-check'), ('litellm_proxy_master_key')
SQL
}

zeroed_daily_condition() {
	cat <<'SQL'
ABS(spend) < 1e-12
  AND COALESCE(prompt_tokens, 0) = 0
  AND COALESCE(completion_tokens, 0) = 0
  AND COALESCE(cache_read_input_tokens, 0) = 0
  AND COALESCE(cache_creation_input_tokens, 0) = 0
  AND COALESCE(api_requests, 0) = 0
  AND COALESCE(successful_requests, 0) = 0
  AND COALESCE(failed_requests, 0) = 0
SQL
}

print_staging_sql() {
	cat <<'SQL'
CREATE TEMP TABLE _litellm_usage_noise_values(value text PRIMARY KEY) ON COMMIT DROP;
INSERT INTO _litellm_usage_noise_values(value)
VALUES
SQL
	internal_noise_values_sql
	cat <<'SQL'
;

CREATE TEMP TABLE _litellm_failure_spendlogs ON COMMIT DROP AS
WITH daily_endpoint_mapping AS MATERIALIZED (
  SELECT key AS call_type, value AS endpoint
  FROM jsonb_each_text(:'route_endpoint_mapping'::jsonb)
)
SELECT
  request_id,
  "startTime",
  metadata,
  ("startTime" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai')::date::text AS date,
  "user",
  team_id,
  end_user,
  request_tags,
  api_key,
  model,
  custom_llm_provider,
  mcp_namespaced_tool_name,
  COALESCE(m.endpoint, '') AS endpoint,
  spend,
  prompt_tokens,
  completion_tokens
FROM "LiteLLM_SpendLogs" s
LEFT JOIN daily_endpoint_mapping m ON m.call_type = s.call_type
WHERE s.status = 'failure';

CREATE TEMP TABLE _litellm_usage_noise_spendlogs ON COMMIT DROP AS
WITH internal_noise_dates AS MATERIALIZED (
  SELECT DISTINCT date FROM "LiteLLM_DailyUserSpend" WHERE api_key IN (SELECT value FROM _litellm_usage_noise_values)
  UNION
  SELECT DISTINCT date FROM "LiteLLM_DailyTeamSpend" WHERE api_key IN (SELECT value FROM _litellm_usage_noise_values)
  UNION
  SELECT DISTINCT date FROM "LiteLLM_DailyTagSpend" WHERE api_key IN (SELECT value FROM _litellm_usage_noise_values) OR tag IN (SELECT value FROM _litellm_usage_noise_values)
)
SELECT DISTINCT s.request_id, s."startTime", s.metadata
FROM internal_noise_dates d
CROSS JOIN LATERAL (
  SELECT s.request_id, s."startTime", s.metadata, s."user", s.api_key
  FROM "LiteLLM_SpendLogs" s
  WHERE s."startTime" >= d.date::timestamp - interval '1 day'
    AND s."startTime" < d.date::timestamp + interval '2 days'
) s
WHERE s.metadata->>'user_api_key_alias' IN (SELECT value FROM _litellm_usage_noise_values)
   OR s.metadata->>'service_account_id' IN (SELECT value FROM _litellm_usage_noise_values)
   OR s."user" IN (SELECT value FROM _litellm_usage_noise_values)
   OR s.api_key IN (SELECT value FROM _litellm_usage_noise_values);

CREATE TEMP TABLE _litellm_usage_noise_cold_storage_targets ON COMMIT DROP AS
WITH deleted_logs AS (
  SELECT request_id, "startTime", metadata FROM _litellm_failure_spendlogs
  UNION ALL
  SELECT request_id, "startTime", metadata FROM _litellm_usage_noise_spendlogs
)
SELECT DISTINCT
  NULLIF(d.metadata->>'cold_storage_object_key', '') AS object_key,
  CASE
    WHEN COALESCE(d.metadata->>'cold_storage_object_key', '') = '' THEN to_char(d."startTime", 'YYYY-MM-DD')
  END AS object_date,
  CASE
    WHEN COALESCE(d.metadata->>'cold_storage_object_key', '') = '' THEN replace(d.request_id, '/', '_')
  END AS object_id
FROM deleted_logs d
WHERE d.request_id <> '';

CREATE TEMP TABLE _litellm_usage_noise_autorouter_sessions ON COMMIT DROP AS
SELECT DISTINCT
  s.api_key,
  left(s.session_id, 512) AS session_id,
  COALESCE(
    NULLIF(s.metadata->'routing_decision'->>'router_model_name', ''),
    NULLIF(s.model_group, '')
  ) AS router_name
FROM "LiteLLM_SpendLogs" s
JOIN _litellm_usage_noise_spendlogs n USING (request_id)
WHERE s.status = 'success'
  AND COALESCE(s.metadata->>'internal_call_origin', '') = ''
  AND s.api_key IS NOT NULL
  AND s.api_key <> ''
  AND s.session_id IS NOT NULL
  AND s.session_id <> ''
  AND s.model IS NOT NULL
  AND s.model <> ''
  AND COALESCE(
    NULLIF(s.metadata->'routing_decision'->>'router_model_name', ''),
    NULLIF(s.model_group, '')
  ) IS NOT NULL;

CREATE TEMP TABLE _litellm_failure_daily_candidates (
  table_name text NOT NULL,
  request_id text NOT NULL,
  entity_value text NOT NULL,
  association_ordinality bigint NOT NULL,
  daily_id text NOT NULL
) ON COMMIT DROP;

INSERT INTO _litellm_failure_daily_candidates
SELECT 'LiteLLM_DailyUserSpend', n.request_id, COALESCE(n."user", ''), 0, d.id
FROM _litellm_failure_spendlogs n
JOIN "LiteLLM_DailyUserSpend" d
  ON COALESCE(d.user_id, '') = COALESCE(n."user", '')
 AND d.date = n.date
 AND d.api_key = n.api_key
 AND COALESCE(d.model, '') = COALESCE(n.model, '')
 AND COALESCE(d.custom_llm_provider, '') = COALESCE(n.custom_llm_provider, '')
 AND COALESCE(d.mcp_namespaced_tool_name, '') = COALESCE(n.mcp_namespaced_tool_name, '')
 AND COALESCE(d.endpoint, '') = n.endpoint
 AND d.failed_requests > 0;

INSERT INTO _litellm_failure_daily_candidates
SELECT 'LiteLLM_DailyTeamSpend', n.request_id, n.team_id, 0, d.id
FROM _litellm_failure_spendlogs n
JOIN "LiteLLM_DailyTeamSpend" d
  ON d.team_id = n.team_id
 AND d.date = n.date
 AND d.api_key = n.api_key
 AND COALESCE(d.model, '') = COALESCE(n.model, '')
 AND COALESCE(d.custom_llm_provider, '') = COALESCE(n.custom_llm_provider, '')
 AND COALESCE(d.mcp_namespaced_tool_name, '') = COALESCE(n.mcp_namespaced_tool_name, '')
 AND COALESCE(d.endpoint, '') = n.endpoint
 AND d.failed_requests > 0
WHERE n.team_id IS NOT NULL;

INSERT INTO _litellm_failure_daily_candidates
SELECT 'LiteLLM_DailyEndUserSpend', n.request_id, n.end_user, 0, d.id
FROM _litellm_failure_spendlogs n
JOIN "LiteLLM_DailyEndUserSpend" d
  ON d.end_user_id = n.end_user
 AND d.date = n.date
 AND d.api_key = n.api_key
 AND COALESCE(d.model, '') = COALESCE(n.model, '')
 AND COALESCE(d.custom_llm_provider, '') = COALESCE(n.custom_llm_provider, '')
 AND COALESCE(d.mcp_namespaced_tool_name, '') = COALESCE(n.mcp_namespaced_tool_name, '')
 AND COALESCE(d.endpoint, '') = n.endpoint
 AND d.failed_requests > 0
WHERE n.end_user IS NOT NULL AND n.end_user <> '';

INSERT INTO _litellm_failure_daily_candidates
SELECT 'LiteLLM_DailyTagSpend', n.request_id, tag.value, tag.ordinality, d.id
FROM _litellm_failure_spendlogs n
CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(n.request_tags, '[]'::jsonb))
  WITH ORDINALITY AS tag(value, ordinality)
JOIN "LiteLLM_DailyTagSpend" d
  ON d.tag = tag.value
 AND d.date = n.date
 AND d.api_key = n.api_key
 AND COALESCE(d.model, '') = COALESCE(n.model, '')
 AND COALESCE(d.custom_llm_provider, '') = COALESCE(n.custom_llm_provider, '')
 AND COALESCE(d.mcp_namespaced_tool_name, '') = COALESCE(n.mcp_namespaced_tool_name, '')
 AND COALESCE(d.endpoint, '') = n.endpoint
 AND d.failed_requests > 0;
SQL
}

print_dry_run_sql() {
	cat <<'SQL'
BEGIN;
SQL
	print_staging_sql
	cat <<'SQL'

SELECT 'LiteLLM_SpendLogs failed requests' AS item, COUNT(*) AS rows_to_change
FROM _litellm_failure_spendlogs
UNION ALL
SELECT 'LiteLLM_SpendLogs internal health/master key noise', COUNT(*)
FROM _litellm_usage_noise_spendlogs
UNION ALL
SELECT 'MinIO cold-storage targets associated with deleted logs', COUNT(*)
FROM _litellm_usage_noise_cold_storage_targets
UNION ALL
SELECT 'LiteLLM_AutoRouterSession orphaned by internal noise cleanup', COUNT(*)
FROM "LiteLLM_AutoRouterSession" a
JOIN _litellm_usage_noise_autorouter_sessions n USING (api_key, session_id, router_name)
WHERE NOT EXISTS (
  SELECT 1
  FROM "LiteLLM_SpendLogs" s
  WHERE s.request_id NOT IN (SELECT request_id FROM _litellm_usage_noise_spendlogs)
    AND s.status = 'success'
    AND COALESCE(s.metadata->>'internal_call_origin', '') = ''
    AND s.api_key = a.api_key
    AND left(s.session_id, 512) = a.session_id
    AND COALESCE(
      NULLIF(s.metadata->'routing_decision'->>'router_model_name', ''),
      NULLIF(s.model_group, '')
    ) = a.router_name
    AND s.model IS NOT NULL
    AND s.model <> ''
)
SQL

	for table in "${DAILY_TABLES[@]}"; do
		if [[ "$table" == "LiteLLM_DailyTagSpend" ]]; then
			match_label="failed-request tag associations"
		else
			match_label="failed requests matched to spend logs"
		fi
		cat <<SQL
UNION ALL
SELECT '${table} ${match_label}', COUNT(DISTINCT (request_id, entity_value, association_ordinality))
FROM _litellm_failure_daily_candidates
WHERE table_name = '${table}'
SQL
		cat <<SQL
UNION ALL
SELECT '${table} rows with failed-request aggregates', COUNT(*)
FROM "${table}"
WHERE COALESCE(failed_requests, 0) > 0
SQL
		if [[ "$table" == "LiteLLM_DailyTagSpend" ]]; then
			cat <<SQL
UNION ALL
SELECT '${table} internal health/master key noise', COUNT(*)
FROM "${table}"
WHERE api_key IN (SELECT value FROM _litellm_usage_noise_values)
   OR tag IN (SELECT value FROM _litellm_usage_noise_values)
SQL
		else
			cat <<SQL
UNION ALL
SELECT '${table} internal health/master key noise', COUNT(*)
FROM "${table}"
WHERE api_key IN (SELECT value FROM _litellm_usage_noise_values)
SQL
		fi
	done

	cat <<'SQL'
UNION ALL
SELECT 'LiteLLM_DailyGatewayRequests rows with failed-request aggregates', COUNT(*)
FROM "LiteLLM_DailyGatewayRequests"
WHERE failed_requests > 0
UNION ALL
SELECT 'LiteLLM_DailyGatewayRequests rows containing only failed requests', COUNT(*)
FROM "LiteLLM_DailyGatewayRequests"
WHERE successful_requests = 0 AND failed_requests > 0
;
ROLLBACK;
SQL
}

print_cleanup_sql() {
	cat <<'SQL'
BEGIN;
LOCK TABLE "LiteLLM_DailyUserSpend", "LiteLLM_DailyTeamSpend", "LiteLLM_DailyTagSpend", "LiteLLM_DailyEndUserSpend", "LiteLLM_DailyGatewayRequests" IN SHARE ROW EXCLUSIVE MODE;
SQL
	print_staging_sql
	cat <<'SQL'

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM _litellm_failure_daily_candidates
    GROUP BY table_name, request_id, entity_value
    HAVING COUNT(*) > 1
  ) THEN
    RAISE EXCEPTION 'A failed request matched multiple daily rows; aborting cleanup';
  END IF;
END
$$;

DO $$
DECLARE
  target record;
  updated_rows integer;
  current_api_requests bigint;
  current_failed_requests bigint;
BEGIN
  FOR target IN
    SELECT
      c.table_name,
      c.daily_id,
      COUNT(*) AS request_count,
      COALESCE(SUM(n.spend), 0) AS spend,
      COALESCE(SUM(n.prompt_tokens), 0) AS prompt_tokens,
      COALESCE(SUM(n.completion_tokens), 0) AS completion_tokens
    FROM _litellm_failure_daily_candidates c
    JOIN _litellm_failure_spendlogs n USING (request_id)
    GROUP BY c.table_name, c.daily_id
  LOOP
    EXECUTE format(
      'SELECT api_requests, failed_requests FROM %I WHERE id = $1 FOR UPDATE',
      target.table_name
    ) INTO current_api_requests, current_failed_requests USING target.daily_id;
    IF current_api_requests IS NULL OR current_failed_requests IS NULL THEN
      RAISE EXCEPTION 'Matched daily row disappeared during cleanup; aborting cleanup';
    END IF;
    IF current_failed_requests < target.request_count THEN
      RAISE EXCEPTION 'Daily row has fewer failed requests than matched failure logs; aborting cleanup';
    END IF;
    IF current_api_requests < target.request_count THEN
      RAISE EXCEPTION 'Daily row has fewer API requests than matched failure logs; aborting cleanup';
    END IF;
    EXECUTE format(
      'UPDATE %I SET spend = CASE WHEN api_requests = $4 THEN 0 ELSE GREATEST(spend - $1, 0) END, prompt_tokens = GREATEST(prompt_tokens - $2, 0), completion_tokens = GREATEST(completion_tokens - $3, 0), failed_requests = failed_requests - $4, api_requests = api_requests - $4 WHERE id = $5',
      target.table_name
    ) USING target.spend, target.prompt_tokens, target.completion_tokens, target.request_count, target.daily_id;
    GET DIAGNOSTICS updated_rows = ROW_COUNT;
    IF updated_rows <> 1 THEN
      RAISE EXCEPTION 'Matched daily row disappeared during cleanup; aborting cleanup';
    END IF;
  END LOOP;
END
$$;

DELETE FROM "LiteLLM_SpendLogs"
WHERE request_id IN (
  SELECT request_id
  FROM _litellm_failure_spendlogs
);

DELETE FROM "LiteLLM_SpendLogs"
WHERE request_id IN (
  SELECT request_id
  FROM _litellm_usage_noise_spendlogs
);

DELETE FROM "LiteLLM_AutoRouterSession" a
USING _litellm_usage_noise_autorouter_sessions n
WHERE a.api_key = n.api_key
  AND a.session_id = n.session_id
  AND a.router_name = n.router_name
  AND NOT EXISTS (
    SELECT 1
    FROM "LiteLLM_SpendLogs" s
    WHERE s.status = 'success'
      AND COALESCE(s.metadata->>'internal_call_origin', '') = ''
      AND s.api_key = a.api_key
      AND left(s.session_id, 512) = a.session_id
      AND COALESCE(
        NULLIF(s.metadata->'routing_decision'->>'router_model_name', ''),
        NULLIF(s.model_group, '')
      ) = a.router_name
      AND s.model IS NOT NULL
      AND s.model <> ''
  );

SQL

	for table in "${DAILY_TABLES[@]}"; do
		cat <<SQL

UPDATE "${table}"
SET spend = CASE WHEN successful_requests = 0 THEN 0 ELSE spend END,
    prompt_tokens = CASE WHEN successful_requests = 0 THEN 0 ELSE prompt_tokens END,
    completion_tokens = CASE WHEN successful_requests = 0 THEN 0 ELSE completion_tokens END,
    cache_read_input_tokens = CASE WHEN successful_requests = 0 THEN 0 ELSE cache_read_input_tokens END,
    cache_creation_input_tokens = CASE WHEN successful_requests = 0 THEN 0 ELSE cache_creation_input_tokens END,
    api_requests = successful_requests,
    failed_requests = 0
WHERE failed_requests > 0;

DELETE FROM "${table}"
WHERE ($(zeroed_daily_condition));

SQL
		if [[ "$table" == "LiteLLM_DailyTagSpend" ]]; then
			cat <<SQL

DELETE FROM "${table}"
WHERE api_key IN (SELECT value FROM _litellm_usage_noise_values)
   OR tag IN (SELECT value FROM _litellm_usage_noise_values);
SQL
		else
			cat <<SQL

DELETE FROM "${table}"
WHERE api_key IN (SELECT value FROM _litellm_usage_noise_values);
SQL
		fi
	done

	cat <<'SQL'

UPDATE "LiteLLM_DailyGatewayRequests"
SET failed_requests = 0
WHERE failed_requests > 0;

DELETE FROM "LiteLLM_DailyGatewayRequests"
WHERE successful_requests = 0 AND failed_requests = 0;

SELECT concat_ws('|', COALESCE(object_key, ''), COALESCE(object_date, ''), COALESCE(object_id, ''))
FROM _litellm_usage_noise_cold_storage_targets
ORDER BY object_key, object_date, object_id;

COMMIT;
SQL
}

if [[ "$ACTION" == "--dry-run" ]]; then
	print_dry_run_sql | "${PSQL[@]}"
	exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始清理 usage noise..."

SUMMARY=$(print_dry_run_sql | "${PSQL[@]}" -t -A -F '|')

COLD_STORAGE_TARGETS_FILE=$(mktemp "${TMPDIR:-/tmp}/litellm-cleanup-cold-storage-targets.XXXXXX")
COLD_STORAGE_OBJECTS_FILE=$(mktemp "${TMPDIR:-/tmp}/litellm-cleanup-cold-storage-objects.XXXXXX")
if ! print_cleanup_sql | "${PSQL[@]}" -t -A >"$COLD_STORAGE_TARGETS_FILE"; then
	rm -f "$COLD_STORAGE_TARGETS_FILE" "$COLD_STORAGE_OBJECTS_FILE"
	exit 1
fi

S3_BUCKET_NAME="${S3_BUCKET_NAME:-litellm-spend-details}"
declare -A COLD_STORAGE_IDS=()
declare -A COLD_STORAGE_DATES=()
while IFS='|' read -r object_key object_date object_id; do
	if [[ -n "$object_key" ]]; then
		printf 'local/%s/%s\n' "$S3_BUCKET_NAME" "$object_key" >>"$COLD_STORAGE_OBJECTS_FILE"
	elif [[ -n "$object_date" && -n "$object_id" ]]; then
		COLD_STORAGE_IDS["$object_date/$object_id"]=1
		COLD_STORAGE_DATES["$object_date"]=1
	fi
done <"$COLD_STORAGE_TARGETS_FILE"

for object_date in "${!COLD_STORAGE_DATES[@]}"; do
	COLD_STORAGE_DATE_OBJECTS_FILE=$(mktemp "${TMPDIR:-/tmp}/litellm-cleanup-cold-storage-date.XXXXXX")
	if ! "${COMPOSE[@]}" exec -T minio mc find "local/$S3_BUCKET_NAME/$object_date" --name '*.json' \
		>"$COLD_STORAGE_DATE_OBJECTS_FILE"; then
		rm -f "$COLD_STORAGE_DATE_OBJECTS_FILE"
		echo "数据库清理已完成，但 MinIO 对象匹配失败。目标清单保留在 $COLD_STORAGE_TARGETS_FILE" >&2
		exit 1
	fi
	while IFS= read -r object_path; do
		object_name="${object_path##*/}"
		object_id="${object_name#time-??-??-??-??????_}"
		object_id="${object_id%.json}"
		if [[ -n "${COLD_STORAGE_IDS["$object_date/$object_id"]+x}" ]]; then
			printf '%s\n' "$object_path" >>"$COLD_STORAGE_OBJECTS_FILE"
		fi
	done <"$COLD_STORAGE_DATE_OBJECTS_FILE"
	rm -f "$COLD_STORAGE_DATE_OBJECTS_FILE"
done

sort -u -o "$COLD_STORAGE_OBJECTS_FILE" "$COLD_STORAGE_OBJECTS_FILE"
if [[ -s "$COLD_STORAGE_OBJECTS_FILE" ]] && \
	! "${COMPOSE[@]}" exec -T minio mc rm --stdin --force <"$COLD_STORAGE_OBJECTS_FILE"; then
	echo "数据库清理已完成，但 MinIO 对象删除失败。对象清单保留在 $COLD_STORAGE_OBJECTS_FILE" >&2
	exit 1
fi
rm -f "$COLD_STORAGE_TARGETS_FILE" "$COLD_STORAGE_OBJECTS_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 清理完成。各项目预估影响条数："
while IFS='|' read -r item rows; do
	[[ -z "$item" ]] && continue
	printf '  %-55s %s\n' "$item" "$rows"
done <<<"$SUMMARY"
