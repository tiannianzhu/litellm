import re
import uuid
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value, string_value
from psycopg import Connection, sql
from pydantic import JsonValue

from tests.integration._support.database import isolated_schema, read_rows

COST_PER_REQUEST: Final = 20 * 0.001 + 20 * 0.002


@pytest.fixture
def postgresql_connection() -> Iterator[Connection[tuple[object, ...]]]:
    with isolated_schema() as database:
        yield database.connection


def _logged(key: str, requests: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT model, to_char("startTime", \'YYYY-MM-DD\') AS day FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
            (sha256(key.encode()).hexdigest(),),
        ),
        lambda rows: len(rows) == requests,
        seconds=70,
    )


def _team_entries(report: JsonValue, day: str, team_names: frozenset[str]) -> dict[str, dict[str, JsonValue]]:
    assert isinstance(report, list), report
    days: Final = [
        object_value(row) for row in report if string_value(object_value(row)["group_by_day"]).startswith(day)
    ]
    assert len(days) == 1, report
    teams: Final = days[0]["teams"]
    assert isinstance(teams, list)
    return {
        string_value(object_value(team)["team_name"]): object_value(team)
        for team in teams
        if object_value(team)["team_name"] in team_names
    }


def test_default_report_groups_each_days_spend_by_team_with_per_key_breakdown(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        busy_alias: Final = f"integration-{uuid.uuid4().hex}"
        quiet_alias: Final = f"integration-{uuid.uuid4().hex}"
        busy: Final = scenario.team(team_alias=busy_alias, models=[model])
        quiet: Final = scenario.team(team_alias=quiet_alias, models=[model])
        busy_key: Final = scenario.key(team_id=busy, models=[model])
        quiet_key: Final = scenario.key(team_id=quiet, models=[model])
        traffic: Final = tuple(
            gateway.chat(model, key=key, text=f"report {uuid.uuid4().hex}") for key in (busy_key, busy_key, quiet_key)
        )
        assert len({response["id"] for response in traffic}) == 3
        busy_rows: Final = _logged(busy_key, 2)
        _logged(quiet_key, 1)
        day: Final = string_value(busy_rows[0]["day"])
        stored_model: Final = busy_rows[0]["model"]
        response: Final = gateway.request("GET", "/global/spend/report", params={"start_date": day, "end_date": day})
        assert response.status_code == 200, response.text
        entries: Final = _team_entries(response.json(), day, frozenset({busy_alias, quiet_alias}))
        assert sorted(entries) == sorted((busy_alias, quiet_alias))
        assert float(str(entries[busy_alias]["total_spend"])) == pytest.approx(2 * COST_PER_REQUEST)
        assert float(str(entries[quiet_alias]["total_spend"])) == pytest.approx(COST_PER_REQUEST)
        breakdown: Final = entries[busy_alias]["metadata"]
        assert isinstance(breakdown, list)
        assert [
            (entry["model"], entry["api_key"], float(str(entry["spend"])), entry["total_tokens"])
            for entry in map(object_value, breakdown)
        ] == [(stored_model, sha256(busy_key.encode()).hexdigest(), pytest.approx(2 * COST_PER_REQUEST), 80)]
        filtered: Final = gateway.request(
            "GET", "/global/spend/report", params={"start_date": day, "end_date": day, "team_id": quiet}
        )
        assert filtered.status_code == 200, filtered.text
        only: Final = filtered.json()
        assert len(only) == 1 and [object_value(team)["team_name"] for team in only[0]["teams"]] == [quiet_alias], only


@pytest.mark.parametrize(
    ("message", "error_class", "error_code", "expected"),
    (
        (
            "Cannot connect to host worker.invalid:9000 ssl:default [socket unavailable]",
            "InternalServerError",
            "500",
            "backend_unavailable",
        ),
        ("Connect call failed ('192.0.2.1', 9000)", "InternalServerError", "500", "backend_unavailable"),
        (
            "Cannot connect to host worker.invalid:9000 [Connection timed out]",
            "InternalServerError",
            "500",
            "backend_timeout",
        ),
        ("Provider returned an internal error", "InternalServerError", "500", "backend_5xx"),
        (
            "hosted_vllm Chat bridge cannot replay encrypted reasoning; use plaintext reasoning content",
            "BadRequestError",
            "400",
            "protocol_capability_limit",
        ),
        (
            "Unsupported chat content part type: 'file'. Supported types: text",
            "BadRequestError",
            "400",
            "file_adaptation_unverified",
        ),
        ("Unsupported reasoning effort 'unknown-level'.", "UnsupportedParamsError", "400", "unsupported_parameter"),
        (
            "Conflicting output token limits: max_tokens=400, max_completion_tokens=200",
            "BadRequestError",
            "400",
            "client_request_format",
        ),
        (
            "This model always has reasoning enabled and cannot be disabled.",
            "UnsupportedParamsError",
            "400",
            "expected_policy",
        ),
        ("Exceeded max_agentic_loops=3", "AgenticLoopLimitError", "", "expected_policy"),
        ("Transaction already closed", "TransactionExpiredError", "", "gateway_database"),
        ("key lookup did not answer within 0.5s", "DBLookupDeadlineExceeded", "", "gateway_database"),
        (
            '504: {"error_code":"P2028","message":"Unable to start a transaction in the given time."}',
            "EngineRequestError",
            "",
            "gateway_database",
        ),
        (
            '500: {"error_code": "P1017", "message":"Server has closed the connection."}',
            "EngineRequestError",
            "",
            "gateway_database",
        ),
        ("Provider returned an internal error", "EngineRequestError", "500", "backend_5xx"),
        ("", "ReadTimeout", "", "backend_timeout"),
        ("", "WriteTimeout", "", "backend_timeout"),
        ("", "ConnectTimeout", "", "backend_timeout"),
        (
            "Request decoded media exceeds the serving limit",
            "BadRequestError",
            "400",
            "client_media_limit",
        ),
        (
            "custom tool arguments must include a content field",
            "BadGatewayError",
            "502",
            "backend_invalid_response",
        ),
        ("body.tools validation errors", "BadRequestError", "400", "client_request_format"),
        (
            "max_tokens=700 exceeds the configured output token limit of 500",
            "BadRequestError",
            "400",
            "client_context_limit",
        ),
    ),
)
def test_spendlogs_report_classifies_failure_causes(
    postgresql_connection: Connection[tuple[object, ...]],
    message: str,
    error_class: str,
    error_code: str,
    expected: str,
) -> None:
    script: Final = (
        Path(__file__).resolve().parents[3] / "litellm/scripts/skills/inspect-errors/scripts/spendlogs_report.sh"
    ).read_text()
    classification: Final = re.search(
        r"SELECT \$classification\$(.*?)\$classification\$ AS classification_sql", script, re.DOTALL
    )
    assert classification is not None
    result: Final = postgresql_connection.execute(
        sql.SQL(
            "WITH failures(message, error_class, error_code) AS "
            "(VALUES ({}::text, {}::text, {}::text)) SELECT {} FROM failures"
        ).format(
            sql.Literal(message), sql.Literal(error_class), sql.Literal(error_code), sql.SQL(classification.group(1))
        ),
    ).fetchone()
    assert result == (expected,)


@pytest.mark.parametrize(("recorded_failures", "expected_state"), ((2, "consistent"), (3, "mismatch")))
def test_spendlogs_report_reconciles_call_failures_by_request_start_day(
    postgresql_connection: Connection[tuple[object, ...]], recorded_failures: int, expected_state: str
) -> None:
    script: Final = (
        Path(__file__).resolve().parents[3] / "litellm/scripts/skills/inspect-errors/scripts/spendlogs_report.sh"
    ).read_text()
    reconciliation: Final = re.search(r"\\echo 'CALL_FAILURE_RECONCILIATION'\n(.*?)\n\\echo", script, re.DOTALL)
    assert reconciliation is not None
    postgresql_connection.execute(
        'CREATE TEMP TABLE "LiteLLM_SpendLogs" ("startTime" timestamp, "endTime" timestamp, status text); '
        'CREATE TEMP TABLE "LiteLLM_DailyGlobalSpend" (date text, failed_requests bigint); '
        'CREATE TEMP TABLE "LiteLLM_DailyRequestErrors" (date text, failed_requests bigint); '
        'INSERT INTO "LiteLLM_SpendLogs" VALUES '
        "('2026-11-08 15:59:00', '2026-11-08 16:01:00', 'failure'), "
        "('2026-11-08 10:00:00', '2026-11-08 10:00:01', 'failure'), "
        "('2026-11-08 10:00:00', '2026-11-08 10:00:01', 'success')"
    )
    for table in ("LiteLLM_DailyGlobalSpend", "LiteLLM_DailyRequestErrors"):
        postgresql_connection.execute(
            sql.SQL("INSERT INTO {} VALUES (%s, %s)").format(sql.Identifier(table)),
            ("2026-11-08", recorded_failures),
        )
    rows: Final = postgresql_connection.execute(sql.SQL(reconciliation.group(1))).fetchall()
    assert rows == [("2026-11-08", recorded_failures, recorded_failures, 2, recorded_failures - 2, expected_state)]
