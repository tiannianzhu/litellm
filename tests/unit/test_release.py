import os
import shutil
import subprocess
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
TEST_VERSION: Final = "1.2.3"
TEST_REVISION: Final = "a" * 40
COMPOSE_PREFIX: Final = "compose --env-file .env.production -f docker-compose.yml -f docker-compose.prod.yml"


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
    (repo_root / "pyproject.toml").write_text(f'[project]\nversion = "{TEST_VERSION}"\n')

    calls_path = tmp_path / "docker-calls.txt"
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$DOCKER_CALLS"\n'
        'if [[ "$*" == *"config --quiet" ]]; then\n'
        '    exit "${DOCKER_CONFIG_STATUS:-0}"\n'
        "fi\n"
    )
    fake_docker.chmod(0o755)

    fake_git: Final = bin_dir / "git"
    fake_git.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "{TEST_REVISION}"\n')
    fake_git.chmod(0o755)

    env: Final = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DOCKER_CALLS": str(calls_path),
        "DOCKER_CONFIG_STATUS": "0",
        "LITELLM_VERSION": "",
    }
    return script, calls_path, env


@pytest.mark.parametrize("services", [(), ("litellm",), ("litellm", "db")])
def test_restart_only_restarts_existing_services(tmp_path: Path, services: tuple[str, ...]) -> None:
    script, calls_path, env = _release_script(tmp_path)

    result: Final = subprocess.run(
        [script, "restart", *services], cwd=script.parent.parent, env=env, text=True, capture_output=True, check=True
    )

    assert calls_path.read_text().splitlines() == [
        "compose version",
        f"{COMPOSE_PREFIX} restart" + (" " + " ".join(services) if services else ""),
    ]
    assert f"==> Restarting version {TEST_VERSION}" + (f": {' '.join(services)}" if services else "") in result.stdout
    assert "==> Done." in result.stdout


@pytest.mark.parametrize("pull", [False, True])
def test_release_only_refreshes_images_when_requested(tmp_path: Path, pull: bool) -> None:
    script, calls_path, env = _release_script(tmp_path)

    result: Final = subprocess.run(
        [script, *(["--pull"] if pull else [])],
        cwd=script.parent.parent,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert calls_path.read_text().splitlines() == [
        "compose version",
        f"{COMPOSE_PREFIX} config --quiet",
        f"{COMPOSE_PREFIX} build{' --pull' if pull else ''} litellm",
        *([f"{COMPOSE_PREFIX} pull --policy always --ignore-buildable"] if pull else []),
        f"{COMPOSE_PREFIX} up -d --remove-orphans db",
        f"{COMPOSE_PREFIX} run --rm --no-deps -e DISABLE_SCHEMA_UPDATE=false litellm"
        " --config=/app/config.yaml --skip_server_startup",
        f"{COMPOSE_PREFIX} up -d --no-build --remove-orphans",
    ]
    assert f"==> Done. Released {TEST_VERSION} ({TEST_REVISION}) on the local host." in result.stdout


def test_check_only_validates_compose_configuration(tmp_path: Path) -> None:
    script, calls_path, env = _release_script(tmp_path)
    result: Final = subprocess.run([script, "check"], env=env, text=True, capture_output=True, check=True)

    assert calls_path.read_text().splitlines() == ["compose version", f"{COMPOSE_PREFIX} config --quiet"]
    assert "==> Check passed." in result.stdout
    assert "==> Done. Released" not in result.stdout


@pytest.mark.parametrize("args", [(), ("--pull",), ("check",)])
def test_invalid_compose_configuration_stops_before_deployment(tmp_path: Path, args: tuple[str, ...]) -> None:
    script, calls_path, env = _release_script(tmp_path)
    result: Final = subprocess.run(
        [script, *args], env={**env, "DOCKER_CONFIG_STATUS": "1"}, text=True, capture_output=True
    )

    assert result.returncode == 1
    assert calls_path.read_text().splitlines() == ["compose version", f"{COMPOSE_PREFIX} config --quiet"]
    assert "==> Check passed." not in result.stdout
    assert "==> Done. Released" not in result.stdout


@pytest.mark.parametrize("args", [("--unknown",), ("--pull", "litellm"), ("litellm", "db")])
def test_invalid_arguments_fail_before_docker(tmp_path: Path, args: tuple[str, ...]) -> None:
    script, calls_path, env = _release_script(tmp_path)
    result: Final = subprocess.run([script, *args], env=env, text=True, capture_output=True)

    assert result.returncode == 1
    assert "Usage:" in result.stderr
    assert not calls_path.exists()
