**In total litellm runs 1000+ tests**

[02/20/2025] Update:

To make it easier to contribute and map what behavior is tested,

we've started mapping the litellm directory in `tests/unit`

This folder can only run mock tests.

Database tests require an existing test PostgreSQL service configured through `DATABASE_URL`. Export it or load the repository's ignored `.env` with `uv run --env-file .env --no-sync pytest <test paths>`

Small SQL suites use `isolated_schema()` from `tests/integration/_support/database.py` through module-local fixtures. It provides a connection and schema-scoped psycopg/Prisma URLs, then removes the schema during teardown, including after failures

Proxy behavior and migration suites keep their existing database setup and isolation requirements. Point `DATABASE_URL` at a test database; sharing a server does not make concurrent writes to one schema safe
