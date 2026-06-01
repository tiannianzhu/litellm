import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast
from urllib.parse import parse_qsl, quote, urlencode, urlsplit
from uuid import uuid4

import psycopg
import pytest
from dotenv import dotenv_values
from psycopg import pq, sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from pytest_postgresql import factories

_ENV_FILE: Final = Path(__file__).resolve().parents[2] / ".env"
_LIBPQ_PARAMETERS: Final = frozenset(parameter.keyword.decode() for parameter in pq.Conninfo.get_defaults())
_postgresql_proc: Final = factories.postgresql_proc()
_postgresql_local: Final = factories.postgresql("_postgresql_proc")


@dataclass(frozen=True, slots=True)
class PostgresTestDatabase:
    connection: psycopg.Connection[tuple[object, ...]] = field(repr=False)
    dsn: str = field(repr=False)
    url: str = field(repr=False)
    schema: str


def configured_database_url(environment: Mapping[str, str], env_file: Path = _ENV_FILE) -> str | None:
    if "DATABASE_URL" in environment:
        return environment["DATABASE_URL"] or None
    if environment.get("PYTHON_DOTENV_DISABLED", "").lower() in ("1", "true", "t", "yes", "y"):
        return None
    return dotenv_values(env_file).get("DATABASE_URL")


def psycopg_dsn(database_url: str, schema: str) -> str:
    parsed: Final = urlsplit(database_url)
    parameters: Final = tuple(
        (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key in _LIBPQ_PARAMETERS
    )
    compatible_url: Final = parsed._replace(query=urlencode(parameters)).geturl()
    options: Final = conninfo_to_dict(compatible_url).get("options", "")
    return make_conninfo(compatible_url, options=f"{options} -csearch_path={schema}")


def prisma_url(database_url: str, schema: str) -> str:
    parsed: Final = urlsplit(database_url)
    parameters: Final = {
        **dict(parse_qsl(parsed.query, keep_blank_values=True)),
        "schema": schema,
        "connection_limit": "1",
    }
    return parsed._replace(query=urlencode(parameters)).geturl()


@contextmanager
def isolated_database(database_url: str) -> Iterator[PostgresTestDatabase]:
    schema: Final = "litellm_test_" + uuid4().hex
    dsn: Final = psycopg_dsn(database_url, schema)
    with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            with psycopg.connect(dsn, connect_timeout=5) as connection:
                yield PostgresTestDatabase(connection, dsn, prisma_url(database_url, schema), schema)
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def _connection_url(connection: psycopg.Connection[tuple[object, ...]]) -> str:
    info: Final = connection.info
    host: Final = f"[{info.host}]" if ":" in info.host else info.host
    return (
        f"postgresql://{quote(info.user, safe='')}:{quote(info.password, safe='')}@{host}:{info.port}/"
        f"{quote(info.dbname, safe='')}"
    )


@pytest.fixture
def postgresql_database(request: pytest.FixtureRequest) -> Iterator[PostgresTestDatabase]:
    configured: Final = configured_database_url(os.environ)
    if configured is not None:
        with isolated_database(configured) as database:
            yield database
        return

    local: Final = cast(psycopg.Connection[tuple[object, ...]], request.getfixturevalue("_postgresql_local"))
    with isolated_database(_connection_url(local)) as database:
        yield database


@pytest.fixture
def postgresql_connection(postgresql_database: PostgresTestDatabase) -> psycopg.Connection[tuple[object, ...]]:
    return postgresql_database.connection
