#### Analytics Endpoints #####
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Final

import fastapi
from fastapi import APIRouter, Depends, HTTPException, status

from litellm.proxy._types import *
from litellm.proxy.analytics_endpoints.cache_activity import CacheActivityResponse, get_cache_activity
from litellm.proxy.auth.authorization import resolve_spend_log_read_scope
from litellm.proxy.auth.authorization_dependencies import LogTeamLookupDependency
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router: Final = APIRouter()


def _parse_bound(value: str, param_name: str, *, end: bool = False) -> str:
    try:
        day: Final = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    else:
        if not end:
            return day.isoformat()
        if day.date() == date.max:
            return "10000-01-01T00:00:00+00:00"
        return (day + timedelta(days=1)).isoformat()

    try:
        timestamp: Final = datetime.fromisoformat(f"{value[:-1]}+00:00" if value.endswith("Z") else value)
        if timestamp.tzinfo is not None:
            return timestamp.isoformat()
    except ValueError:
        pass
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": f"{param_name} must be a YYYY-MM-DD date or a timestamp with a timezone, got {value!r}"},
    )


@router.get(
    "/global/activity/cache_hits",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CacheActivityResponse,
    include_in_schema=False,
)
async def get_global_activity(
    log_team_lookup: LogTeamLookupDependency,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    start_date: Annotated[str, fastapi.Query(description="Inclusive UTC date or timestamp with a timezone")],
    end_date: Annotated[str, fastapi.Query(description="Inclusive UTC date or exclusive timestamp with a timezone")],
    key_aliases: Annotated[
        list[str] | None, fastapi.Query(description="Only include spend from these key aliases")
    ] = None,
    models: Annotated[list[str] | None, fastapi.Query(description="Only include spend for these models")] = None,
) -> CacheActivityResponse:
    """
    Cache activity for the Admin UI cache dashboard, aggregated per call_type:
    cache hits vs successful LLM API requests vs failed requests, plus totals
    for the stat cards and the available key-alias/model filter options.

    Admin callers see proxy-wide activity; all other callers see only their own
    spend logs and those of teams they may view, exactly like ``/spend/logs/ui``.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            },
        )

    return await get_cache_activity(
        prisma_client=prisma_client,
        read_scope=await resolve_spend_log_read_scope(user_api_key_dict, log_team_lookup),
        start=_parse_bound(start_date, "start_date"),
        end_exclusive=_parse_bound(end_date, "end_date", end=True),
        key_aliases=key_aliases or [],
        models=models or [],
    )
