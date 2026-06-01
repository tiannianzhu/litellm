#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ $# -gt 1 && "${1:-}" != "restart" ]] || [[ $# -eq 1 && ! "${1:-}" =~ ^(check|restart|--pull)$ ]]; then
	echo "Usage: $0 [--pull]" >&2
	echo "       $0 check" >&2
	echo "       $0 restart [service ...]" >&2
	exit 1
fi

command -v docker >/dev/null 2>&1 || {
	echo "docker is required" >&2
	exit 1
}
docker compose version >/dev/null 2>&1 || {
	echo "docker compose is required" >&2
	exit 1
}
command -v git >/dev/null 2>&1 || {
	echo "git is required" >&2
	exit 1
}
[[ -f .env.production ]] || {
	echo ".env.production is required" >&2
	exit 1
}

resolve_litellm_version() {
	if [[ -n "${LITELLM_VERSION:-}" ]]; then
		printf '%s\n' "${LITELLM_VERSION}"
		return
	fi

	awk '
		/^\[project\]/ { in_project = 1; next }
		/^\[/ && in_project { exit }
		in_project && $1 == "version" {
			gsub(/"/, "", $3)
			print $3
			exit
		}
	' pyproject.toml
}

LITELLM_VERSION="$(resolve_litellm_version)"
if [[ -z "${LITELLM_VERSION}" ]]; then
	echo "failed to resolve LITELLM_VERSION from pyproject.toml" >&2
	exit 1
fi
LITELLM_REVISION="$(git rev-parse HEAD)"
export LITELLM_VERSION LITELLM_REVISION

COMPOSE_ARGS=(--env-file .env.production -f docker-compose.yml -f docker-compose.prod.yml)

PULL_IMAGES=0
if [[ "${1:-}" == "--pull" ]]; then
	PULL_IMAGES=1
	shift
fi

if [[ "${1:-}" == "restart" ]]; then
	shift
	echo "==> Restarting version ${LITELLM_VERSION}${*:+: $*}"
	docker compose "${COMPOSE_ARGS[@]}" restart "$@"
	echo "==> Done."
	exit 0
fi

echo "==> Check compose configuration"
docker compose "${COMPOSE_ARGS[@]}" config --quiet
if [[ "${1:-}" == "check" ]]; then
	echo "==> Check passed."
	exit 0
fi

LITELLM_IMAGE="litellm-local:${LITELLM_VERSION}"

echo "==> Version:       ${LITELLM_VERSION}"
echo "==> Revision:      ${LITELLM_REVISION}"
echo "==> LiteLLM image: ${LITELLM_IMAGE}"
echo "==> Build and deploy on local host"
if ((PULL_IMAGES)); then
	docker compose "${COMPOSE_ARGS[@]}" build --pull litellm
	docker compose "${COMPOSE_ARGS[@]}" pull --policy always --ignore-buildable
else
	docker compose "${COMPOSE_ARGS[@]}" build litellm
fi
echo "==> Run database migrations"
docker compose "${COMPOSE_ARGS[@]}" up -d --remove-orphans db
docker compose "${COMPOSE_ARGS[@]}" run --rm --no-deps -e DISABLE_SCHEMA_UPDATE=false litellm \
	--config=/app/config.yaml \
	--skip_server_startup
docker compose "${COMPOSE_ARGS[@]}" up -d --no-build --remove-orphans

echo "==> Done. Released ${LITELLM_VERSION} (${LITELLM_REVISION}) on the local host."
