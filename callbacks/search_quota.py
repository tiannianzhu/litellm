from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
from typing import Final, LiteralString, Protocol, cast
from zoneinfo import ZoneInfo

from litellm.exceptions import RateLimitError
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import CallTypes

DAILY_LIMIT: Final = 300


class QuotaDatabase(Protocol):
    async def execute_raw(self, query: LiteralString, *args: str | int) -> int: ...


class _ProxyDatabaseClient(Protocol):
    db: QuotaDatabase


def _proxy_database() -> QuotaDatabase:
    from litellm.proxy.proxy_server import prisma_client

    client: Final = cast(_ProxyDatabaseClient | None, prisma_client)
    if client is None:
        raise RuntimeError("Web search quota requires the proxy database")
    return client.db


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _identity(kwargs: Mapping[str, object]) -> str | None:
    params: Final = _mapping(kwargs.get("litellm_params"))
    metadata_sources: Final = tuple(
        _mapping(source)
        for source in (
            kwargs.get("litellm_metadata"),
            kwargs.get("metadata"),
            params.get("metadata"),
            params.get("litellm_metadata"),
        )
    )
    end_user: Final = next(
        (
            value
            for metadata in metadata_sources
            for value in (metadata.get("user_api_key_end_user_id"),)
            if isinstance(value, str) and value.strip()
        ),
        None,
    )
    if end_user is not None:
        return f"user:{end_user}"
    key: Final = next(
        (
            value
            for metadata in metadata_sources
            for value in (metadata.get("user_api_key"),)
            if isinstance(value, str) and value
        ),
        None,
    )
    return f"key:{sha256(key.encode()).hexdigest()}" if key is not None else None


@dataclass(frozen=True, slots=True)
class DailySearchQuota:
    database: QuotaDatabase | None = None

    async def reserve(self, identity: str, day: date, count: int) -> bool:
        if not 1 <= count <= DAILY_LIMIT:
            return False
        database: Final = self.database if self.database is not None else _proxy_database()
        affected: Final = await database.execute_raw(
            'INSERT INTO "LiteLLM_SearchQuota" (day, identity, calls) VALUES ($1::date, $2, $3) '
            'ON CONFLICT (day, identity) DO UPDATE SET calls = "LiteLLM_SearchQuota".calls + EXCLUDED.calls '
            'WHERE "LiteLLM_SearchQuota".calls + EXCLUDED.calls <= $4',
            day.isoformat(),
            identity,
            count,
            DAILY_LIMIT,
        )
        return affected == 1


class SearchQuota(CustomLogger):
    def __init__(self, quota: DailySearchQuota) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger accepts untyped keyword arguments.
        self.quota = quota

    async def async_pre_call_deployment_hook(self, kwargs: dict[str, object], call_type: CallTypes | None) -> None:
        if call_type != CallTypes.asearch:
            return
        identity: Final = _identity(kwargs)
        query: Final = kwargs.get("query")
        count: Final = len(cast(list[object], query)) if isinstance(query, list) else 1
        day: Final = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        if identity is not None and await self.quota.reserve(identity, day, count):
            return
        provider: Final = kwargs.get("search_provider")
        provider_name: Final = provider if isinstance(provider, str) else "search"
        raise RateLimitError(
            message=(
                "Web search daily quota reached (300 searches per user, resets at 00:00 Asia/Shanghai). "
                "Stop searching and use the results already collected."
                if identity is not None
                else "Web search requires an identified end user or API key."
            ),
            llm_provider=provider_name,
            model=f"{provider_name}/search",
            max_retries=0,
        )


proxy_handler_instance: Final = SearchQuota(DailySearchQuota())
