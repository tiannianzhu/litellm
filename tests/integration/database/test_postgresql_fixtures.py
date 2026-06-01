from typing import Final

import pytest

from tests._support.postgresql import PostgresTestDatabase, isolated_database


def test_database_isolation_preserves_other_schemas_and_cleans_up_after_failure(
    postgresql_database: PostgresTestDatabase,
) -> None:
    outer: Final = postgresql_database.connection
    outer.execute("CREATE TABLE fixture_sentinel (value INTEGER)")
    outer.execute("INSERT INTO fixture_sentinel VALUES (42)")
    outer.commit()
    schemas_before: Final = outer.execute("SELECT nspname FROM pg_namespace ORDER BY nspname").fetchall()

    def write_and_fail() -> None:
        with isolated_database(postgresql_database.url) as inner:
            inner.connection.execute("CREATE TABLE fixture_sentinel (value INTEGER)")
            inner.connection.execute("INSERT INTO fixture_sentinel VALUES (7)")
            inner.connection.commit()
            assert inner.connection.execute("SELECT value FROM fixture_sentinel").fetchone() == (7,)
            assert outer.execute("SELECT value FROM fixture_sentinel").fetchone() == (42,)
            raise RuntimeError("controlled fixture failure")

    with pytest.raises(RuntimeError, match="controlled fixture failure"):
        write_and_fail()

    assert outer.execute("SELECT value FROM fixture_sentinel").fetchone() == (42,)
    assert outer.execute("SELECT nspname FROM pg_namespace ORDER BY nspname").fetchall() == schemas_before
