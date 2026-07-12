from collections.abc import Mapping, Sequence
from typing import Final

import httpx
from pydantic import BaseModel, Field

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str


class BochaWebPage(BaseModel):
    name: str = ""
    url: str = ""
    snippet: str = ""
    summary: str = ""
    date_published: str | None = Field(default=None, alias="datePublished")


class BochaWebPages(BaseModel):
    value: tuple[BochaWebPage, ...] = Field(default_factory=tuple)


class BochaSearchData(BaseModel):
    web_pages: BochaWebPages | None = Field(default=None, alias="webPages")


class BochaSearchPayload(BaseModel):
    code: int | str | None = None
    msg: str | None = None
    data: BochaSearchData | None = None


_BOCHA_PROVIDER_PARAMS: Final = frozenset(("freshness", "summary", "include", "exclude"))


class BochaSearchConfig(BaseSearchConfig):
    BOCHA_API_BASE: Final = "https://api.bochaai.com/v1"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Bocha AI"

    def validate_environment(
        self,
        headers: Mapping[str, str],
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig extension contract
    ) -> dict[str, str]:  # mutable-ok: HTTP client contract requires mutable headers
        resolved_api_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("BOCHA_API_KEY",),
            base_env_var="BOCHA_API_BASE",
            default_api_base=self.BOCHA_API_BASE,
        )
        if not resolved_api_key:
            raise ValueError("BOCHA_API_KEY is not set. Set the `BOCHA_API_KEY` environment variable.")
        return {  # mutable-ok: HTTP client requires mutable headers
            **headers,
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: Mapping[str, object],
        data: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig extension contract
    ) -> str:
        resolved_api_base: Final = api_base or get_secret_str("BOCHA_API_BASE") or self.BOCHA_API_BASE
        normalized_api_base: Final = resolved_api_base.rstrip("/")
        if normalized_api_base.endswith("/web-search"):
            return normalized_api_base
        return f"{normalized_api_base}/web-search"

    def transform_search_request(
        self,
        query: str | Sequence[str],
        optional_params: Mapping[str, object],
        **kwargs: object,  # kwargs-ok: BaseSearchConfig extension contract
    ) -> dict[str, object]:  # mutable-ok: provider request contract requires a mutable JSON object
        normalized_query: Final = query if isinstance(query, str) else " ".join(query)
        return {  # mutable-ok: Bocha's HTTP request body must be a JSON object
            "query": normalized_query,
            "count": optional_params.get("max_results", 10),
            **{key: value for key, value in optional_params.items() if key in _BOCHA_PROVIDER_PARAMS},
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig extension contract
    ) -> SearchResponse:
        payload: Final = BochaSearchPayload.model_validate_json(raw_response.text)
        if payload.code not in (None, 200, "200"):
            raise ValueError(f"Bocha Search API returned code={payload.code}: {payload.msg or 'Unknown error'}")
        web_pages: Final = payload.data.web_pages if payload.data is not None else None
        pages: Final = web_pages.value if web_pages is not None else ()
        return SearchResponse(
            results=[  # mutable-ok: SearchResponse contract requires a mutable result list
                SearchResult(
                    title=page.name,
                    url=page.url,
                    snippet=page.summary or page.snippet,
                    date=page.date_published,
                    last_updated=None,
                )
                for page in pages
            ],
            object="search",
        )
