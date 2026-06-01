import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_fake_docker(bin_dir: Path) -> None:
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "$*" == *ROUTE_ENDPOINT_MAPPING* ]]; then\n'
        '  printf "runtime\\n" >> "$EVENTS"\n'
        '  printf \'{"aresponses":"/responses"}\\n\'\n'
        "  exit 0\n"
        "fi\n"
        'if [[ "$*" == *"exec -T minio mc rm"* ]]; then\n'
        '  printf "minio-rm\\n" >> "$EVENTS"\n'
        '  cat > "$REMOVED_KEYS"\n'
        '  [[ "${FAKE_S3_FAIL:-0}" != 1 ]] || exit 1\n'
        "  exit 0\n"
        "fi\n"
        'if [[ "$*" != *"exec -T db psql"* ]]; then\n'
        '  printf "unexpected: %s\\n" "$*" >> "$EVENTS"\n'
        "  exit 99\n"
        "fi\n"
        "payload=$(cat)\n"
        'printf "sql\\n" >> "$EVENTS"\n'
        'printf "\\n-- invocation --\\n" >> "$CAPTURED_SQL"\n'
        'printf "%s\\n" "$payload" >> "$CAPTURED_SQL"\n'
        'if [[ "${FAKE_SQL_FAIL:-0}" == 1 && "$payload" == *"SELECT concat_ws"* ]]; then\n'
        "  exit 1\n"
        "fi\n"
        'if [[ "$payload" == *"SELECT concat_ws"* ]]; then\n'
        "  printf 'direct/probe.json||\\n'\n"
        "fi\n"
    )
    fake_docker.chmod(0o755)


def _script_environment(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    test_root = tmp_path / "repo"
    scripts_dir = test_root / "scripts"
    bin_dir = test_root / "bin"
    scripts_dir.mkdir(parents=True)
    bin_dir.mkdir()

    script = scripts_dir / "cleanup_usage_noise.sh"
    shutil.copy2(REPO_ROOT / "scripts" / script.name, script)
    (test_root / ".env.production").write_text("POSTGRES_USER=test\nPOSTGRES_DB=test\n")
    (test_root / "docker-compose.yml").touch()
    _write_fake_docker(bin_dir)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["TMPDIR"] = str(tmp_path)
    env["CAPTURED_SQL"] = str(tmp_path / "captured.sql")
    env["EVENTS"] = str(tmp_path / "events.txt")
    env["REMOVED_KEYS"] = str(tmp_path / "removed-keys.txt")
    return test_root, script, env


def _events(env: dict[str, str]) -> list[str]:
    return Path(env["EVENTS"]).read_text().splitlines()


DAILY_SUFFIXES = ("User", "Team", "EndUser", "Tag")


def test_preview_does_not_delete_s3(tmp_path: Path) -> None:
    _, script, env = _script_environment(tmp_path)
    subprocess.run([script], env=env, check=True)
    assert _events(env) == ["runtime", "sql"]


def test_apply_deletes_only_explicit_object_paths(tmp_path: Path) -> None:
    _, script, env = _script_environment(tmp_path)
    subprocess.run([script, "--apply"], env=env, check=True)
    assert _events(env) == ["runtime", "sql", "sql", "minio-rm"]
    assert Path(env["REMOVED_KEYS"]).read_text() == "local/litellm-spend-details/direct/probe.json\n"


def test_sql_failure_does_not_delete_s3(tmp_path: Path) -> None:
    _, script, env = _script_environment(tmp_path)
    env["FAKE_SQL_FAIL"] = "1"
    result = subprocess.run([script, "--apply"], env=env, capture_output=True)
    assert result.returncode != 0
    assert _events(env) == ["runtime", "sql", "sql"]


def test_s3_failure_preserves_retry_manifest(tmp_path: Path) -> None:
    _, script, env = _script_environment(tmp_path)
    env["FAKE_S3_FAIL"] = "1"
    result = subprocess.run([script, "--apply"], env=env, text=True, capture_output=True)
    assert result.returncode != 0
    manifest = Path(result.stderr.strip().split("对象清单保留在 ")[-1])
    assert manifest.read_text() == "local/litellm-spend-details/direct/probe.json\n"
    manifest.unlink()


def _fixture_sql() -> str:
    tables = {
        "User": "user_id",
        "Team": "team_id",
        "EndUser": "end_user_id",
        "Tag": "tag",
        "Organization": "organization_id",
        "Agent": "agent_id",
    }
    sql = """
CREATE TEMP TABLE "LiteLLM_EndUserTable" (user_id text PRIMARY KEY, spend double precision, budget_id text);
INSERT INTO "LiteLLM_EndUserTable" VALUES ('visitor', 10, NULL);
CREATE TEMP TABLE "LiteLLM_AutoRouterSession" (api_key text, session_id text, router_name text);
CREATE TEMP TABLE "LiteLLM_DailyGatewayRequests" (date text, route text, successful_requests bigint, failed_requests bigint, updated_at timestamp);
INSERT INTO "LiteLLM_DailyGatewayRequests" VALUES ('2020-01-01','old',0,1,'2020-01-01'), ('2020-01-01','recent',5,1,now());
CREATE TEMP TABLE "LiteLLM_Config" (param_name text, param_value jsonb);
CREATE TEMP TABLE "LiteLLM_SpendLogs" (
    request_id text PRIMARY KEY, call_type text DEFAULT 'aresponses', api_key text DEFAULT 'health-probe',
    team_id text DEFAULT 'health-probe', "user" text DEFAULT '', end_user text DEFAULT 'visitor',
    organization_id text DEFAULT 'org', agent_id text DEFAULT 'agent', model text DEFAULT 'model',
    model_group text DEFAULT '', custom_llm_provider text DEFAULT '', mcp_namespaced_tool_name text DEFAULT '', session_id text DEFAULT '',
    status text DEFAULT 'failure', spend double precision DEFAULT 0, total_tokens int DEFAULT 0,
    prompt_tokens int DEFAULT 0, completion_tokens int DEFAULT 0,
    "startTime" timestamp DEFAULT '2020-01-01', "endTime" timestamp DEFAULT '2020-01-01',
    updated_at timestamp DEFAULT (now() - interval '2 minutes'), request_tags jsonb DEFAULT '["health-probe"]',
    metadata jsonb DEFAULT '{"cold_storage_object_key":"probe.json","usage_object":{}}'
);
INSERT INTO "LiteLLM_SpendLogs" (request_id) VALUES ('probe');
"""
    for suffix in DAILY_SUFFIXES:
        entity = tables[suffix]
        value = {
            "User": "",
            "Team": "health-probe",
            "EndUser": "visitor",
            "Tag": "health-probe",
            "Organization": "org",
            "Agent": "agent",
        }[suffix]
        sql += f"""
CREATE TEMP TABLE "LiteLLM_Daily{suffix}Spend" (
    id text PRIMARY KEY, {entity} text, date text DEFAULT '2020-01-01', api_key text DEFAULT 'health-probe',
    updated_at timestamp DEFAULT '2020-01-01', model text DEFAULT 'model', custom_llm_provider text DEFAULT '', mcp_namespaced_tool_name text DEFAULT '',
    endpoint text DEFAULT '/responses', api_requests bigint DEFAULT 1, successful_requests bigint DEFAULT 0,
    failed_requests bigint DEFAULT 1, spend double precision DEFAULT 0, prompt_tokens bigint DEFAULT 0,
    completion_tokens bigint DEFAULT 0, cache_read_input_tokens bigint DEFAULT 0,
    cache_creation_input_tokens bigint DEFAULT 0, compression_saved_tokens bigint DEFAULT 0,
    compression_savings_spend double precision DEFAULT 0, prompt_caching_savings_spend double precision DEFAULT 0,
    gateway_injected_caching_savings_spend double precision DEFAULT 0, autorouter_savings_spend double precision DEFAULT 0
);
INSERT INTO "LiteLLM_Daily{suffix}Spend" (id, {entity}) VALUES ('daily', '{value}');
"""
    return sql


def _scenario_sql(scenario: str) -> str:
    if scenario == "mixed":
        return """
UPDATE "LiteLLM_SpendLogs" SET spend=0.5, prompt_tokens=4, total_tokens=4, metadata='{}';
INSERT INTO "LiteLLM_SpendLogs" (request_id, status, spend, prompt_tokens, total_tokens, metadata)
VALUES ('paid', 'success', 2, 10, 10, '{"cold_storage_object_key":"paid.json"}');
""" + "".join(
            f"""UPDATE "LiteLLM_Daily{suffix}Spend" SET api_requests=2, successful_requests=1,
            spend=2.5, prompt_tokens=14, cache_read_input_tokens=3;"""
            for suffix in DAILY_SUFFIXES
        )
    if scenario == "recent":
        return """
INSERT INTO "LiteLLM_SpendLogs" (request_id, prompt_tokens, total_tokens, "endTime", updated_at)
VALUES ('recent',5,5,now(),now());
""" + "".join(
            f"""UPDATE "LiteLLM_Daily{suffix}Spend" SET api_requests=2, failed_requests=2,
            prompt_tokens=5, updated_at=now();"""
            for suffix in DAILY_SUFFIXES
        )
    if scenario == "internal":
        return "UPDATE \"LiteLLM_SpendLogs\" SET api_key='litellm_proxy_master_key', status='success';" + "".join(
            f"""UPDATE "LiteLLM_Daily{suffix}Spend" SET api_key='litellm_proxy_master_key',
            successful_requests=1, failed_requests=0;"""
            for suffix in DAILY_SUFFIXES
        )
    return 'UPDATE "LiteLLM_DailyTeamSpend" SET api_requests=0;'


@pytest.mark.skipif(not os.getenv("LITELLM_CLEANUP_TEST_POSTGRES_CONTAINER"), reason="requires PostgreSQL container")
@pytest.mark.parametrize("scenario", ["mixed", "recent", "internal", "insufficient-count"])
def test_cleanup_sql_contract(tmp_path: Path, scenario: str) -> None:
    _, script, env = _script_environment(tmp_path)
    subprocess.run([script, "--apply"], env=env, capture_output=True, check=True)
    sql = Path(env["CAPTURED_SQL"]).read_text()
    command = [
        "docker",
        "exec",
        "-i",
        os.environ["LITELLM_CLEANUP_TEST_POSTGRES_CONTAINER"],
        "psql",
        "-U",
        os.getenv("LITELLM_CLEANUP_TEST_POSTGRES_USER", "postgres"),
        "-qAt",
        "-v",
        "ON_ERROR_STOP=1",
        "-v",
        'route_endpoint_mapping={"aresponses":"/responses"}',
        "-v",
        "max_end_user_budget_id=",
    ]
    checks_sql = "".join(
        f"""SELECT 'daily:' || COALESCE(SUM(api_requests),0) || ':' || COALESCE(SUM(successful_requests),0)
        || ':' || COALESCE(SUM(failed_requests),0) FROM "LiteLLM_Daily{suffix}Spend";
        SELECT 'metrics:' || COALESCE(SUM(spend),0) || ':' || COALESCE(SUM(prompt_tokens),0)
        || ':' || COALESCE(SUM(cache_read_input_tokens),0) FROM "LiteLLM_Daily{suffix}Spend";"""
        for suffix in DAILY_SUFFIXES
    )
    result = subprocess.run(
        command,
        input=_fixture_sql()
        + _scenario_sql(scenario)
        + sql
        + sql
        + checks_sql
        + """
SELECT 'remaining:' || COUNT(*) FROM "LiteLLM_SpendLogs";
SELECT 'balance:' || spend FROM "LiteLLM_EndUserTable";
SELECT 'gateway:' || route || ':' || failed_requests || ':' || successful_requests FROM "LiteLLM_DailyGatewayRequests";
""",
        text=True,
        capture_output=True,
    )
    if scenario == "insufficient-count":
        assert result.returncode != 0
        assert "fewer API requests" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    remaining, daily, metrics, balance = {
        "mixed": (1, "1:1:0", "2:10:3", "9.5"),
        "recent": (1, "1:0:1", "0:5:0", "10"),
        "internal": (0, "0:0:0", "0:0:0", "10"),
    }[scenario]
    assert f"remaining:{remaining}" in result.stdout
    assert result.stdout.splitlines().count(f"daily:{daily}") == 4
    assert result.stdout.splitlines().count(f"metrics:{metrics}") == 4
    assert f"balance:{balance}" in result.stdout
    assert "gateway:recent:0:5" in result.stdout
    assert "gateway:old:" not in result.stdout
