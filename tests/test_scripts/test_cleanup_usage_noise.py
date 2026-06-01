import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_help_does_not_require_database_configuration(tmp_path: Path) -> None:
    script = tmp_path / "cleanup_usage_noise.sh"
    shutil.copy2(REPO_ROOT / "scripts" / script.name, script)

    result = subprocess.run(
        [script, "--help"], cwd=tmp_path, text=True, capture_output=True, check=True
    )

    assert result.stdout.startswith("Usage:")


def test_extra_arguments_are_rejected_before_running_docker(tmp_path: Path) -> None:
    script = tmp_path / "cleanup_usage_noise.sh"
    shutil.copy2(REPO_ROOT / "scripts" / script.name, script)

    result = subprocess.run(
        [script, "--dry-run", "extra"], cwd=tmp_path, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert result.stderr.startswith("Usage:")


def test_failed_requests_match_daily_rows_that_hold_failure_counts(tmp_path: Path) -> None:
    test_root = tmp_path / "repo"
    scripts_dir = test_root / "scripts"
    bin_dir = test_root / "bin"
    scripts_dir.mkdir(parents=True)
    bin_dir.mkdir()

    script = scripts_dir / "cleanup_usage_noise.sh"
    shutil.copy2(REPO_ROOT / "scripts" / script.name, script)
    (test_root / ".env.production").write_text("POSTGRES_USER=test\nPOSTGRES_DB=test\n")
    (test_root / "docker-compose.yml").touch()

    captured_sql = tmp_path / "captured.sql"
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *ROUTE_ENDPOINT_MAPPING* ]]; then\n'
        "  printf '{\"aresponses\":\"/responses\"}\\n'\n"
        "  exit 0\n"
        "fi\n"
        'if [[ "$*" == *"exec -T minio mc rm"* ]]; then\n'
        '  printf "%s\\n" "$*" > "$MINIO_COMMAND"\n'
        '  cat > "$REMOVED_KEYS"\n'
        "  exit 0\n"
        "fi\n"
        'if [[ "$*" == *"exec -T minio mc find"* ]]; then\n'
        "  printf 'local/litellm-spend-details/2026-08-20/time-12-00-00-000000_failure-id.json\\n'\n"
        "  printf 'local/litellm-spend-details/2026-08-20/time-12-00-01-000000_other-id.json\\n'\n"
        "  exit 0\n"
        "fi\n"
        "payload=$(cat)\n"
        'printf "\\n-- invocation --\\n" >> "$CAPTURED_SQL"\n'
        'printf "%s\\n" "$payload" >> "$CAPTURED_SQL"\n'
        'if [[ "$payload" == *"SELECT concat_ws"* ]]; then\n'
        "  printf 'direct/failure.json||\\n'\n"
        "  printf '|2026-08-20|failure-id\\n'\n"
        "fi\n"
    )
    fake_docker.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CAPTURED_SQL"] = str(captured_sql)
    env["MINIO_COMMAND"] = str(tmp_path / "minio-command.txt")
    env["REMOVED_KEYS"] = str(tmp_path / "removed-keys.txt")

    subprocess.run([script, "--apply"], cwd=test_root, env=env, check=True)

    sql = captured_sql.read_text()
    apply_sql = sql.rsplit("-- invocation --", maxsplit=1)[-1]
    daily_lock = (
        'LOCK TABLE "LiteLLM_DailyUserSpend", "LiteLLM_DailyTeamSpend", '
        '"LiteLLM_DailyTagSpend", "LiteLLM_DailyEndUserSpend", '
        '"LiteLLM_DailyGatewayRequests" IN SHARE ROW EXCLUSIVE MODE;'
    )
    assert daily_lock in apply_sql
    assert apply_sql.index(daily_lock) < apply_sql.index(
        "CREATE TEMP TABLE _litellm_usage_noise_values"
    )
    assert sql.count("status = 'failure'") == 2
    assert "api_base IS NULL" not in sql
    assert '"LiteLLM_VerificationToken"' not in sql
    assert "user_api_key_request_route" not in sql
    assert sql.count("jsonb_each_text(:'route_endpoint_mapping'::jsonb)") == 2
    assert sql.count("AND d.failed_requests > 0") == 8
    assert sql.count("AND COALESCE(d.endpoint, '') = n.endpoint") == 8
    assert sql.count("AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai'") == 2
    assert '"startTime"::date::text AS date' not in sql
    assert "current_failed_requests < target.request_count" in sql
    assert "CASE WHEN api_requests = $4 THEN 0" in sql
    assert "ABS(spend) < 1e-12" in sql
    assert "CREATE TEMP TABLE _litellm_usage_noise_autorouter_sessions" in sql
    assert "CREATE TEMP TABLE _litellm_usage_noise_cold_storage_targets" in sql
    assert "NULLIF(d.metadata->>'cold_storage_object_key', '')" in sql
    assert "s.request_id NOT IN (SELECT request_id FROM deleted_request_ids)" not in sql
    assert "SELECT concat_ws('|'," in sql
    assert "left(s.session_id, 512) AS session_id" in sql
    assert (
        'DELETE FROM "LiteLLM_AutoRouterSession" a\n'
        "USING _litellm_usage_noise_autorouter_sessions n" in sql
    )
    assert 'UPDATE "LiteLLM_DailyGatewayRequests"\nSET failed_requests = 0' in sql
    assert (
        'DELETE FROM "LiteLLM_DailyGatewayRequests"\n'
        "WHERE successful_requests = 0 AND failed_requests = 0" in sql
    )
    assert "exec -T minio mc rm --stdin --force" in Path(
        env["MINIO_COMMAND"]
    ).read_text()
    assert "exec -T minio mc rm --stdin --force local/" not in Path(
        env["MINIO_COMMAND"]
    ).read_text()
    assert Path(env["REMOVED_KEYS"]).read_text() == (
        "local/litellm-spend-details/2026-08-20/time-12-00-00-000000_failure-id.json\n"
        "local/litellm-spend-details/direct/failure.json\n"
    )
