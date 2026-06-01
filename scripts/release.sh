#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

cd "${repo_root}"

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

if ! docker compose version >/dev/null 2>&1; then
	echo "docker compose is not available" >&2
	exit 1
fi

for required_file in docker-compose.yml docker-compose.prod.yml .env.production; do
	if [[ ! -f "${required_file}" ]]; then
		echo "missing ${required_file}" >&2
		exit 1
	fi
done

LITELLM_VERSION="$(resolve_litellm_version)"
if [[ -z "${LITELLM_VERSION}" ]]; then
	echo "failed to resolve LITELLM_VERSION from pyproject.toml" >&2
	exit 1
fi
export LITELLM_VERSION
echo "Using LITELLM_VERSION=${LITELLM_VERSION}"

compose_args=(--env-file .env.production -f docker-compose.yml -f docker-compose.prod.yml)

if [[ "${1:-}" == "restart" ]]; then
	shift
	if [[ $# -eq 0 ]]; then
		set -- litellm
	fi
	echo "Restarting existing service(s): $*"
	exec docker compose "${compose_args[@]}" restart "$@"
fi

docker compose "${compose_args[@]}" build --pull litellm
docker compose "${compose_args[@]}" pull --policy always --ignore-buildable
docker compose "${compose_args[@]}" up -d --remove-orphans db
docker compose "${compose_args[@]}" run --rm --no-deps -e DISABLE_SCHEMA_UPDATE=false litellm \
	--config=/app/config.yaml \
	--skip_server_startup \
	--use_v2_migration_resolver \
	--enforce_prisma_migration_check
exec docker compose "${compose_args[@]}" up -d --no-build --remove-orphans "$@"
