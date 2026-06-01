import os
import shutil
import subprocess
from pathlib import Path


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


def test_restart_only_restarts_the_existing_litellm_container(tmp_path: Path) -> None:
    script, calls_path, env = _release_script(tmp_path)

    result = subprocess.run(
        [script, "restart"], cwd=script.parent.parent, env=env, text=True, capture_output=True, check=True
    )

    assert calls_path.read_text().splitlines() == [
        "compose version",
        "compose --env-file .env.production -f docker-compose.yml -f docker-compose.prod.yml restart litellm",
    ]
    assert "Restarting existing service(s): litellm" in result.stdout


def test_release_refreshes_build_and_service_images(tmp_path: Path) -> None:
    script, calls_path, env = _release_script(tmp_path)

    subprocess.run([script], cwd=script.parent.parent, env=env, text=True, capture_output=True, check=True)

    calls = calls_path.read_text().splitlines()
    assert "compose --env-file .env.production -f docker-compose.yml -f docker-compose.prod.yml build --pull litellm" in calls
    assert (
        "compose --env-file .env.production -f docker-compose.yml -f docker-compose.prod.yml pull --policy always --ignore-buildable"
        in calls
    )
