from pathlib import Path
from typing import Final
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

import pytest
from psycopg.conninfo import conninfo_to_dict

from tests._support.postgresql import configured_database_url, prisma_url, psycopg_dsn


@pytest.mark.parametrize(
    ("environment", "expected"),
    (
        ({}, "postgresql://file-user@file-host/file-db"),
        ({"DATABASE_URL": "postgresql://env-user@env-host/env-db"}, "postgresql://env-user@env-host/env-db"),
        ({"DATABASE_URL": ""}, None),
        ({"PYTHON_DOTENV_DISABLED": "1"}, None),
    ),
)
def test_database_configuration_respects_environment_and_dotenv_isolation(
    environment: dict[str, str],
    expected: str | None,
    tmp_path: Path,
) -> None:
    env_file: Final = tmp_path / ".env"
    env_file.write_text("DATABASE_URL=postgresql://file-user@file-host/file-db\n")
    assert configured_database_url(environment, env_file) == expected


def test_database_urls_preserve_authentication_and_driver_options() -> None:
    password: Final = "password:@/?%+"
    options: Final = "-cstatement_timeout=1000"
    query: Final = urlencode(
        {"schema": "old_schema", "connection_limit": "5", "sslmode": "require", "options": options}
    )
    original: Final = f"postgresql://user:{quote(password, safe='')}@localhost:12345/database?{query}"
    schema: Final = "isolated_schema"

    psycopg_parameters: Final = conninfo_to_dict(psycopg_dsn(original, schema))
    prisma_parsed: Final = urlsplit(prisma_url(original, schema))
    prisma_parameters: Final = dict(parse_qsl(prisma_parsed.query))

    assert psycopg_parameters["password"] == password
    assert psycopg_parameters["options"] == f"{options} -csearch_path={schema}"
    assert psycopg_parameters["sslmode"] == prisma_parameters["sslmode"]
    assert prisma_parsed.netloc == urlsplit(original).netloc
    assert prisma_parameters["schema"] == schema
    assert prisma_parameters["connection_limit"] == "1"
