from datetime import datetime, timedelta
from typing import Final
from uuid import uuid4

import litellm
import pytest

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.team_endpoints import get_team_spend_by_user

from tests.integration._support.actors import Actor
from tests.integration._support.prisma_clients import connected_proxy_prisma

pytestmark = pytest.mark.asyncio(loop_scope="session")


# GET /team/spend/by_user shares the team-scope resolver with
# /team/daily/activity, so the membership matrix must hold here too. team_ids
# is mandatory on this route (a per-user rollup with no team is meaningless),
# so the bare query is 400 for everyone instead of defaulting to own teams.
_MEMBERS = {
    "alpha": {
        Actor.TEAM_ADMIN,
        Actor.INTERNAL_USER,
        Actor.OWNER,
        Actor.UNRELATED_SAME_ORG,
        Actor.SERVICE_ACCOUNT,
    },
    "beta": {Actor.CROSS_ORG_USER},
}


def _expected(actor: Actor, team: str) -> int:
    if team == "none":
        return 400
    if actor == Actor.PROXY_ADMIN:
        return 200
    return 200 if actor in _MEMBERS.get(team, set()) else 404


_CASES = [
    (f"{team}/{actor.value}", actor, team, _expected(actor, team))
    for team in ("none", "alpha", "beta")
    for actor in Actor
]

_DATES = "start_date=2024-01-01&end_date=2024-12-31"


@pytest.mark.parametrize(
    "actor,team,expected_status",
    [(a, t, s) for (_id, a, t, s) in _CASES],
    ids=[c[0] for c in _CASES],
)
async def test_team_spend_by_user_matrix(actor: Actor, team: str, expected_status: int, gateway_client, actors):
    team_id = {"alpha": actors.team_alpha_id, "beta": actors.team_beta_id}.get(team)
    query = _DATES if team_id is None else f"{_DATES}&team_ids={team_id}"

    resp = await gateway_client.get(
        f"/team/spend/by_user?{query}",
        headers={"Authorization": f"Bearer {actors.keys[actor].cleartext}"},
    )
    assert resp.status_code == expected_status, f"{actor.value} -> {team}: {resp.status_code} {resp.text}"
    if expected_status == 200:
        body = resp.json()
        assert (body["start_date"], body["end_date"]) == ("2024-01-01", "2024-12-31")
        assert all(row["team_id"] == team_id for row in body["results"])


@pytest.mark.parametrize(
    "usage_timezone,day,utc_start,utc_end",
    (
        ("UTC", "2026-03-08", "2026-03-08T00:00:00", "2026-03-09T00:00:00"),
        ("Asia/Shanghai", "2026-03-08", "2026-03-07T16:00:00", "2026-03-08T16:00:00"),
        ("America/New_York", "2026-03-08", "2026-03-08T05:00:00", "2026-03-09T04:00:00"),
    ),
)
async def test_team_spend_by_user_business_day_boundaries(
    monkeypatch: pytest.MonkeyPatch, usage_timezone: str, day: str, utc_start: str, utc_end: str
) -> None:
    team_id: Final = f"spend-window-{uuid4().hex}"
    start: Final = datetime.fromisoformat(utc_start)
    end: Final = datetime.fromisoformat(utc_end)
    records: Final = (
        ("before", start - timedelta(seconds=1), 8.0),
        ("start", start, 0.25),
        ("inside", end - timedelta(seconds=1), 0.5),
        ("end", end, 16.0),
    )
    monkeypatch.setattr(litellm, "timezone", usage_timezone, raising=False)
    async with connected_proxy_prisma() as client:
        monkeypatch.setattr(proxy_server, "prisma_client", client)
        await client.db.litellm_teamtable.create(data={"team_id": team_id})
        try:
            await client.db.litellm_spendlogs.create_many(
                data=[
                    {
                        "request_id": f"{team_id}-{suffix}",
                        "call_type": "acompletion",
                        "team_id": team_id,
                        "user": team_id,
                        "startTime": at,
                        "endTime": at,
                        "spend": spend,
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                        "status": "success",
                    }
                    for suffix, at, spend in records
                ]
            )
            response: Final = await get_team_spend_by_user(
                user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
                team_ids=team_id,
                start_date=day,
                end_date=day,
            )
            assert (response.start_date, response.end_date) == (day, day)
            assert [
                (row.team_id, row.user_id, row.spend, row.api_requests, row.total_tokens) for row in response.results
            ] == [(team_id, team_id, 0.75, 2, 4)]
        finally:
            await client.db.litellm_spendlogs.delete_many(where={"team_id": team_id})
            await client.db.litellm_teamtable.delete(where={"team_id": team_id})
