import asyncio
import os
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit

import psycopg
import pytest
from prisma import Prisma
from psycopg import sql

from callbacks.openwebui_alias import OpenWebUIAlias
from integration._support.client import Gateway
from integration._support.database import read_rows
from litellm._logging import verbose_proxy_logger
from litellm.proxy import proxy_server
from tests._support.postgresql import PostgresTestDatabase
from tests.integration._support.prisma_clients import connected_prisma


@dataclass(frozen=True, slots=True)
class _AliasDatabase:
    db: Prisma


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", (False, True), ids=("concurrent-create", "preserve-existing"))
async def test_concurrent_end_user_alias_sync_preserves_accounting(
    postgresql_database: PostgresTestDatabase,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    existing: bool,
) -> None:
    connection: Final = postgresql_database.connection
    connection.execute("""
        CREATE TABLE "LiteLLM_EndUserTable" (
            user_id TEXT PRIMARY KEY, alias TEXT, spend DOUBLE PRECISION NOT NULL DEFAULT 0,
            blocked BOOLEAN NOT NULL DEFAULT false, budget_id TEXT, object_permission_id TEXT,
            allowed_model_region TEXT, default_model TEXT, models TEXT[] NOT NULL DEFAULT '{}'
        )
    """)
    if existing:
        connection.execute("""
            INSERT INTO "LiteLLM_EndUserTable" (user_id, alias, spend, blocked, budget_id)
            VALUES ('alias-user', 'Old Alias', 7, true, 'retained-budget')
        """)
    connection.commit()
    parsed: Final = urlsplit(postgresql_database.url)
    url: Final = parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "connection_limit": "2"})).geturl()
    now: Final = datetime(2026, 1, 1)
    kwargs: Final = {
        "litellm_params": {
            "metadata": {"user_api_key_end_user_id": "alias-user"},
            "proxy_server_request": {"headers": {"x-openwebui-user-name": "New Alias"}},
        }
    }
    verbose_proxy_logger.addHandler(caplog.handler)
    try:
        async with connected_prisma(url) as database:
            monkeypatch.setattr(proxy_server, "prisma_client", _AliasDatabase(database))
            connection.execute('LOCK TABLE "LiteLLM_EndUserTable" IN SHARE MODE')
            calls: Final = asyncio.gather(
                *(OpenWebUIAlias().async_log_success_event(kwargs, {}, now, now) for _ in range(2))
            )
            try:
                async with (
                    await psycopg.AsyncConnection.connect(postgresql_database.dsn, autocommit=True) as observer,
                    asyncio.timeout(10),
                ):
                    while True:
                        cursor: Final = await observer.execute(
                            "SELECT COUNT(*) FROM pg_locks WHERE relation = %s::regclass "
                            "AND mode = 'RowExclusiveLock' AND NOT granted",
                            (f'"{postgresql_database.schema}"."LiteLLM_EndUserTable"',),
                        )
                        if await cursor.fetchone() == (2,):
                            break
                        await asyncio.sleep(0.01)
            finally:
                connection.rollback()
                await asyncio.wait_for(calls, timeout=10)
            row: Final = await database.litellm_endusertable.find_unique(where={"user_id": "alias-user"})
            assert row is not None
            assert row.alias == "New Alias"
            assert (row.spend, row.blocked, row.budget_id) == (
                (7.0, True, "retained-budget") if existing else (0.0, False, None)
            )
            assert not [record for record in caplog.records if record.levelname == "ERROR"], caplog.text
    finally:
        verbose_proxy_logger.removeHandler(caplog.handler)


@pytest.mark.covers("other.database.access_group.failed_second_write_rolls_back_first")
def test_access_group_second_key_constraint_failure_rolls_back_all_writes(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        outside: Final = scenario.model()
        keys: Final = (scenario.key(models=[outside]), scenario.key(models=[outside]))
        tokens: Final = [sha256(key.encode()).hexdigest() for key in keys]
        name: Final = f"integration-{uuid.uuid4().hex}"
        constraint: Final = f"integration_reject_{uuid.uuid4().hex}"
        witness: Final = constraint + "_seq"
        check_function: Final = constraint + "_check"
        body: Final = {"access_group_name": name, "access_model_names": [model], "assigned_key_ids": tokens}

        def remove_partial_group() -> None:
            for row in read_rows(
                'SELECT access_group_id FROM "LiteLLM_AccessGroupTable" WHERE access_group_name=%s', (name,)
            ):
                response: Final = gateway.request("DELETE", f"/v1/access_group/{row['access_group_id']}")
                assert response.status_code == 204, response.text
            assert (
                read_rows('SELECT access_group_id FROM "LiteLLM_AccessGroupTable" WHERE access_group_name=%s', (name,))
                == []
            )

        scenario.cleanups.callback(remove_partial_group)
        before: Final = read_rows(
            'SELECT token, access_group_ids FROM "LiteLLM_VerificationToken" WHERE token=ANY(%s) ORDER BY token',
            (tokens,),
        )
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection, ExitStack() as cleanup:
            connection.execute(sql.SQL("CREATE SEQUENCE {}").format(sql.Identifier(witness)))
            connection.execute(sql.SQL("GRANT USAGE ON SEQUENCE {} TO PUBLIC").format(sql.Identifier(witness)))
            cleanup.callback(connection.execute, sql.SQL("DROP SEQUENCE {}").format(sql.Identifier(witness)))
            connection.execute(
                sql.SQL(
                    "CREATE FUNCTION {}(text[]) RETURNS boolean LANGUAGE plpgsql AS $$ BEGIN IF "
                    "cardinality($1)>0 THEN PERFORM nextval({}); RETURN false; END IF; RETURN true; END $$"
                ).format(sql.Identifier(check_function), sql.Literal(witness))
            )
            cleanup.callback(
                connection.execute, sql.SQL("DROP FUNCTION {}(text[])").format(sql.Identifier(check_function))
            )
            connection.execute(
                sql.SQL(
                    'ALTER TABLE "LiteLLM_VerificationToken" ADD '
                    "CONSTRAINT {} CHECK (token <> {} OR {}(access_group_ids))"
                ).format(sql.Identifier(constraint), sql.Literal(tokens[1]), sql.Identifier(check_function))
            )
            cleanup.callback(
                connection.execute,
                sql.SQL('ALTER TABLE "LiteLLM_VerificationToken" DROP CONSTRAINT {}').format(
                    sql.Identifier(constraint)
                ),
            )
            try:
                assert connection.execute(
                    sql.SQL("SELECT is_called FROM {}").format(sql.Identifier(witness))
                ).fetchone() == (False,)
                failed: Final = gateway.request("POST", "/v1/access_group", body)
                assert failed.status_code == 500, failed.text
                assert connection.execute(
                    sql.SQL("SELECT is_called FROM {}").format(sql.Identifier(witness))
                ).fetchone() == (True,)
                assert (
                    read_rows(
                        'SELECT access_group_id FROM "LiteLLM_AccessGroupTable" WHERE access_group_name=%s', (name,)
                    )
                    == []
                )
                assert (
                    read_rows(
                        "SELECT token, access_group_ids FROM "
                        '"LiteLLM_VerificationToken" WHERE token=ANY(%s) ORDER BY token',
                        (tokens,),
                    )
                    == before
                )
                for key in keys:
                    denied: Final = gateway.request(
                        "POST",
                        "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": "rolled back grant"}]},
                        key=key,
                    )
                    assert denied.status_code == 403 and denied.json()["error"]["type"] == "key_model_access_denied", (
                        denied.text
                    )
            finally:
                cleanup.close()
        created: Final = gateway.request("POST", "/v1/access_group", body)
        assert created.status_code == 201, created.text
        identity: Final = created.json()["access_group_id"]
        try:
            for key in keys:
                assert gateway.chat(model, key=key)["usage"]["total_tokens"] == 40
        finally:
            deleted: Final = gateway.request("DELETE", f"/v1/access_group/{identity}")
            assert deleted.status_code == 204, deleted.text
            assert (
                read_rows(
                    'SELECT access_group_id FROM "LiteLLM_AccessGroupTable" WHERE access_group_id=%s', (identity,)
                )
                == []
            )
        assert (
            read_rows(
                'SELECT token, access_group_ids FROM "LiteLLM_VerificationToken" WHERE token=ANY(%s) ORDER BY token',
                (tokens,),
            )
            == before
        )
        assert read_rows("SELECT conname FROM pg_constraint WHERE conname=%s", (constraint,)) == []
