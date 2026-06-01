import os
import shutil
import subprocess
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _release_script(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    bin_dir = repo_root / "bin"
    scripts_dir.mkdir(parents=True)
    bin_dir.mkdir()

    script = scripts_dir / "release.sh"
    shutil.copy2(REPO_ROOT / "scripts" / script.name, script)
    (repo_root / "docker-compose.yml").touch()
    (repo_root / "docker-compose.prod.yml").touch()
    (repo_root / ".env.production").touch()
    (repo_root / "pyproject.toml").write_text('[project]\nversion = "1.100.0"\n')

    calls_path = tmp_path / "docker-calls.txt"
    fake_docker = bin_dir / "docker"
    fake_docker.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$DOCKER_CALLS"\n')
    fake_docker.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["DOCKER_CALLS"] = str(calls_path)
    return script, calls_path, env


@pytest.mark.parametrize("services", [(), ("litellm",), ("litellm", "db")])
def test_restart_only_restarts_existing_services(tmp_path: Path, services: tuple[str, ...]) -> None:
    script, calls_path, env = _release_script(tmp_path)

    result: Final = subprocess.run(
        [script, "restart", *services], cwd=script.parent.parent, env=env, text=True, capture_output=True, check=True
    )

    assert calls_path.read_text().splitlines() == [
        "compose version",
        "compose --env-file .env.production -f docker-compose.yml -f docker-compose.prod.yml restart"
        + (" " + " ".join(services) if services else ""),
    ]
    assert f"Restarting existing service(s): {' '.join(services) if services else 'all'}" in result.stdout


@pytest.mark.parametrize("pull", [False, True])
def test_release_only_refreshes_images_when_requested(tmp_path: Path, pull: bool) -> None:
    script, calls_path, env = _release_script(tmp_path)

    subprocess.run(
        [script, *(["--pull"] if pull else [])],
        cwd=script.parent.parent,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    prefix: Final = "compose --env-file .env.production -f docker-compose.yml -f docker-compose.prod.yml"
    assert calls_path.read_text().splitlines() == [
        "compose version",
        f"{prefix} build{' --pull' if pull else ''} litellm",
        *([f"{prefix} pull --policy always --ignore-buildable"] if pull else []),
        f"{prefix} up -d --remove-orphans db",
        f"{prefix} run --rm --no-deps -e DISABLE_SCHEMA_UPDATE=false litellm"
        " --config=/app/config.yaml --skip_server_startup --enforce_prisma_migration_check",
        f"{prefix} up -d --no-build --remove-orphans",
    ]


@pytest.mark.parametrize("args", [("--unknown",), ("--pull", "litellm"), ("litellm", "db")])
def test_invalid_arguments_fail_before_docker(tmp_path: Path, args: tuple[str, ...]) -> None:
    script, calls_path, env = _release_script(tmp_path)
    result: Final = subprocess.run([script, *args], env=env, text=True, capture_output=True)

    assert result.returncode == 1
    assert "Usage:" in result.stderr
    assert not calls_path.exists()
