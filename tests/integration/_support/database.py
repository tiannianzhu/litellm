import os
import uuid
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final, LiteralString
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
from psycopg import pq, sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from pydantic import JsonValue, TypeAdapter

ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])
_LIBPQ_PARAMETERS: Final = frozenset(parameter.keyword.decode() for parameter in pq.Conninfo.get_defaults())


@dataclass(frozen=True, slots=True)
class PostgresTestDatabase:
    connection: psycopg.Connection[tuple[object, ...]] = field(repr=False)
    dsn: str = field(repr=False)
    url: str = field(repr=False)
    schema: str


@contextmanager
def isolated_schema() -> Iterator[PostgresTestDatabase]:
    schema: Final = "litellm_test_" + uuid.uuid4().hex
    parsed: Final = urlsplit(os.environ["DATABASE_URL"])
    parameters: Final = dict(parse_qsl(parsed.query, keep_blank_values=True))
    compatible_url: Final = parsed._replace(
        query=urlencode({key: value for key, value in parameters.items() if key in _LIBPQ_PARAMETERS})
    ).geturl()
    options: Final = conninfo_to_dict(compatible_url).get("options", "")
    dsn: Final = make_conninfo(compatible_url, options=f"{options} -csearch_path={schema}")
    url: Final = parsed._replace(query=urlencode({**parameters, "schema": schema, "connection_limit": "1"})).geturl()
    with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            with psycopg.connect(dsn, connect_timeout=5) as connection:
                yield PostgresTestDatabase(connection, dsn, url, schema)
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def read_rows(
    query: str, parameters: tuple[str, ...], *, database_url: str | None = None
) -> list[dict[str, JsonValue]]:
    with psycopg.connect(database_url or os.environ["DATABASE_URL"], row_factory=dict_row) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        return ROWS.validate_python(connection.execute(query, parameters).fetchall())


def write_rows(query: LiteralString, parameters: tuple[str, ...], *, database_url: str | None = None) -> None:
    with psycopg.connect(database_url or os.environ["DATABASE_URL"]) as connection:
        connection.execute(query, parameters)


@contextmanager
def scratch_database(prefix: str = "integration") -> Generator[str]:
    name: Final = f"{prefix}_{uuid.uuid4().hex}"
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            yield urlunsplit(urlsplit(os.environ["DATABASE_URL"])._replace(path=f"/{name}"))
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
