import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm


@pytest.mark.asyncio
async def test_bocha_search_request_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOCHA_API_KEY", "test-api-key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = json.dumps(
        {
            "code": 200,
            "data": {
                "webPages": {
                    "value": [
                        {
                            "name": "LiteLLM documentation",
                            "url": "https://docs.litellm.ai/",
                            "snippet": "Call multiple LLM providers through one interface.",
                            "summary": "LiteLLM provides a unified API gateway for LLMs.",
                            "datePublished": "2026-07-10T00:00:00+08:00",
                        }
                    ]
                }
            },
        }
    )

    with patch(  # test-quality-ok: litellm.asearch has no client injection seam
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.asearch(
            query="LiteLLM",
            search_provider="bocha",
            max_results=5,
            freshness="oneMonth",
            summary=True,
        )

    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["url"] == "https://api.bochaai.com/v1/web-search"
    assert call_kwargs["headers"] == {
        "Authorization": "Bearer test-api-key",
        "Content-Type": "application/json",
    }
    assert call_kwargs["json"] == {
        "query": "LiteLLM",
        "count": 5,
        "freshness": "oneMonth",
        "summary": True,
    }
    assert response.object == "search"
    assert len(response.results) == 1
    assert response.results[0].title == "LiteLLM documentation"
    assert response.results[0].url == "https://docs.litellm.ai/"
    assert response.results[0].snippet == "LiteLLM provides a unified API gateway for LLMs."
    assert response.results[0].date == "2026-07-10T00:00:00+08:00"


@pytest.mark.asyncio
async def test_bocha_search_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOCHA_API_KEY", raising=False)

    with pytest.raises(Exception, match="BOCHA_API_KEY is not set"):
        await litellm.asearch(query="LiteLLM", search_provider="bocha")


@pytest.mark.asyncio
async def test_bocha_search_does_not_send_server_key_to_untrusted_api_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOCHA_API_KEY", "server-api-key")

    with pytest.raises(Exception, match="Refusing to send the server-configured BOCHA_API_KEY"):
        await litellm.asearch(
            query="LiteLLM",
            search_provider="bocha",
            api_base="https://example.com/v1",
        )


@pytest.mark.asyncio
async def test_bocha_search_surfaces_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOCHA_API_KEY", "test-api-key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = json.dumps({"code": 403, "msg": "Insufficient balance"})

    with (
        patch(  # test-quality-ok: litellm.asearch has no client injection seam
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
        pytest.raises(Exception, match="Insufficient balance"),
    ):
        await litellm.asearch(query="LiteLLM", search_provider="bocha")
