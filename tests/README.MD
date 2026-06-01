**In total litellm runs 1000+ tests**

[02/20/2025] Update:

To make it easier to contribute and map what behavior is tested,

we've started mapping the litellm directory in `tests/unit`

This folder can only run mock tests.

Database tests use `DATABASE_URL`, either exported or stored in the repository's ignored `.env`. Run them with `uv run --env-file .env --no-sync pytest <test paths>` so suites that read the environment directly receive the same connection

The shared `postgresql_database` and `postgresql_connection` fixtures reuse that PostgreSQL service. Each test creates a unique schema and removes it during teardown, including after failures. The database fixture also provides a schema-scoped Prisma URL and psycopg DSN with the configured credentials. Without a configured URL, these fixtures start a temporary server through `pytest-postgresql`, which requires local PostgreSQL binaries

Proxy behavior and migration suites keep their existing database setup and isolation requirements. Point `DATABASE_URL` at a test database; sharing a server does not make concurrent writes to one schema safe
