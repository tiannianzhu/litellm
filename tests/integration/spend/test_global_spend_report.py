import re
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from psycopg import Connection, sql
from pydantic import JsonValue

COST_PER_REQUEST: Final = 20 * 0.001 + 20 * 0.002


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
    ("message", "expected"),
    (
        ("Cannot connect to host worker.invalid:9000 ssl:default [socket unavailable]", "backend_unavailable"),
        ("Connect call failed ('192.0.2.1', 9000)", "backend_unavailable"),
        ("Cannot connect to host worker.invalid:9000 [Connection timed out]", "backend_timeout"),
        ("Provider returned an internal error", "backend_5xx"),
    ),
)
def test_spendlogs_report_classifies_wrapped_connection_errors(
    postgresql_connection: Connection[tuple[object, ...]], message: str, expected: str
) -> None:
    script: Final = (
        Path(__file__).resolve().parents[3] / ".agents/skills/inspect-errors/scripts/spendlogs_report.sh"
    ).read_text()
    classification: Final = re.search(
        r"SELECT \$classification\$(.*?)\$classification\$ AS classification_sql", script, re.DOTALL
    )
    assert classification is not None
    result: Final = postgresql_connection.execute(
        sql.SQL(
            "WITH failures(message, error_class, error_code) AS "
            "(VALUES ({}::text, 'InternalServerError', '500')) SELECT {} FROM failures"
        ).format(sql.Literal(message), sql.SQL(classification.group(1))),
    ).fetchone()
    assert result == (expected,)
