import asyncio
import base64
import time
from collections.abc import Mapping
from typing import Final

import httpx
import pytest

from callbacks.mineru_document_interceptor import (
    DocumentInterceptionError,
    HttpMinerUDocumentParser,
    MinerUDocumentInterceptor,
    ParseFailure,
    ParseFailureKind,
    ParseSuccess,
)
from litellm.llms.anthropic.messages.adapters.transformation import LiteLLMAnthropicMessagesAdapter


class RecordingParser:
    def __init__(self, markdown: str) -> None:
        self.markdown = markdown
        self.documents: list[bytes] = []

    async def parse(self, filename: str, content: bytes, media_type: str):
        self.documents.append(content)
        return ParseSuccess(markdown=self.markdown)


@pytest.mark.asyncio
async def test_document_block_is_replaced_with_mineru_markdown() -> None:
    parser = RecordingParser("# Extracted\n\nDocument text")
    interceptor = MinerUDocumentInterceptor(parser=parser)
    pdf = base64.b64encode(b"pdf-content").decode()

    result = await interceptor.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data={
            "model": "glm-5.2",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize this"},
                        {
                            "type": "document",
                            "title": "paper.pdf",
                            "context": "Peer-reviewed paper",
                            "cache_control": {"type": "ephemeral"},
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf,
                            },
                        },
                    ],
                }
            ],
        },
        call_type="anthropic_messages",
    )

    assert parser.documents == [b"pdf-content"]
    assert result["messages"][0]["content"] == [
        {"type": "text", "text": "Summarize this"},
        {
            "type": "text",
            "text": (
                '<document title="paper.pdf">\n'
                "<context>Peer-reviewed paper</context>\n"
                "# Extracted\n\nDocument text\n"
                "</document>"
            ),
            "cache_control": {"type": "ephemeral"},
        },
    ]


@pytest.mark.asyncio
async def test_text_document_source_does_not_call_mineru() -> None:
    parser = RecordingParser("unused")
    interceptor = MinerUDocumentInterceptor(parser=parser)

    result = await interceptor.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "Already extracted",
                            },
                        }
                    ],
                }
            ]
        },
        call_type="anthropic_messages",
    )

    assert parser.documents == []
    assert result["messages"][0]["content"] == [{"type": "text", "text": "<document>\nAlready extracted\n</document>"}]


@pytest.mark.asyncio
async def test_invalid_base64_is_rejected_without_echoing_document() -> None:
    parser = RecordingParser("unused")
    interceptor = MinerUDocumentInterceptor(parser=parser)
    invalid_data = "not-valid-base64!!"

    with pytest.raises(DocumentInterceptionError) as exc_info:
        await interceptor.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": invalid_data,
                                },
                            }
                        ],
                    }
                ]
            },
            call_type="anthropic_messages",
        )

    assert exc_info.value.status_code == 400
    assert invalid_data not in str(exc_info.value)


@pytest.mark.asyncio
async def test_non_anthropic_messages_call_is_unchanged() -> None:
    parser = RecordingParser("unused")
    interceptor = MinerUDocumentInterceptor(parser=parser)
    data: dict[str, object] = {"messages": [{"role": "user", "content": "hello"}]}

    result = await interceptor.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=data,
        call_type="acompletion",
    )

    assert result is data
    assert parser.documents == []


@pytest.mark.asyncio
async def test_anthropic_call_without_documents_does_not_require_mineru_settings() -> None:
    interceptor = MinerUDocumentInterceptor()
    data: dict[str, object] = {"messages": [{"role": "user", "content": "hello"}]}

    result = await interceptor.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=data,
        call_type="anthropic_messages",
    )

    assert result is data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind, status, message",
    [
        (ParseFailureKind.TIMEOUT, 504, "Document parsing timed out"),
        (ParseFailureKind.UPSTREAM_FAILURE, 502, "Document parsing service failed"),
    ],
)
async def test_parser_failure_preserves_public_error(kind: ParseFailureKind, status: int, message: str) -> None:
    class FailingParser:
        async def parse(self, filename: str, content: bytes, media_type: str) -> ParseFailure:
            return ParseFailure(kind)

    interceptor: Final = MinerUDocumentInterceptor(parser=FailingParser())
    pdf: Final = base64.b64encode(b"pdf-content").decode()

    with pytest.raises(DocumentInterceptionError) as exc_info:
        await interceptor.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf,
                                },
                            }
                        ],
                    }
                ]
            },
            call_type="anthropic_messages",
        )

    assert exc_info.value.status_code == status
    assert exc_info.value.public_message == message


@pytest.mark.asyncio
async def test_extracted_text_size_limit_applies_to_text_sources() -> None:
    interceptor = MinerUDocumentInterceptor(
        parser=RecordingParser("unused"),
        max_extracted_characters=5,
    )

    with pytest.raises(DocumentInterceptionError) as exc_info:
        await interceptor.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {"type": "text", "data": "too long"},
                            }
                        ],
                    }
                ]
            },
            call_type="anthropic_messages",
        )

    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["completed", "partial"])
async def test_http_parser_uses_async_mineru_task_workflow(terminal_status: str) -> None:
    exchanges: Final = iter(
        (
            (
                "POST",
                "/mineru/v1/tasks",
                httpx.Response(
                    202,
                    json={
                        "task_id": "task-123",
                        "status_url": "/v1/tasks/task-123",
                        "result_url": "/v1/tasks/task-123/result",
                    },
                ),
            ),
            ("GET", "/mineru/v1/tasks/task-123", httpx.Response(200, json={"status": "queued"})),
            ("GET", "/mineru/v1/tasks/task-123", httpx.Response(200, json={"status": "running"})),
            ("GET", "/mineru/v1/tasks/task-123", httpx.Response(200, json={"status": terminal_status})),
            ("GET", "/mineru/v1/tasks/task-123/result", httpx.Response(202, json={"status": "running"})),
            ("GET", "/mineru/v1/tasks/task-123", httpx.Response(200, json={"status": terminal_status})),
            (
                "GET",
                "/mineru/v1/tasks/task-123/result",
                httpx.Response(
                    200,
                    json={
                        "status": terminal_status,
                        "files": [{"status": "completed", "content": {"markdown": "# Parsed"}}],
                    },
                ),
            ),
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        method, path, response = next(exchanges)
        assert (request.method, request.url.path) == (method, path)
        if request.method == "POST":
            assert request.url.params == httpx.QueryParams(
                {"tier": "standard", "ocr_mode": "auto", "page_range": "all", "output_formats": "markdown"}
            )
            assert request.content.count(b"Content-Disposition: form-data;") == 1
            assert b'name="files"; filename="document.pdf"' in request.content
            assert b"Content-Type: application/pdf\r\n\r\npdf\r\n" in request.content
        return response

    parser: Final = HttpMinerUDocumentParser(
        base_url="https://fixture.invalid/mineru/",
        timeout_seconds=5,
        poll_interval_seconds=0,
        transport=httpx.MockTransport(handler),
    )

    assert await parser.parse("document.pdf", b"pdf", "application/pdf") == ParseSuccess(markdown="# Parsed")
    assert tuple(exchanges) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "canceled", "unknown"])
async def test_http_parser_rejects_unsuccessful_tasks(status: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "task-123"})
        assert request.url.path == "/v1/tasks/task-123"
        return httpx.Response(200, json={"status": status})

    parser: Final = HttpMinerUDocumentParser(base_url="https://fixture.invalid", transport=httpx.MockTransport(handler))

    assert await parser.parse("document.pdf", b"pdf", "application/pdf") == ParseFailure(
        ParseFailureKind.UPSTREAM_FAILURE
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "files",
    [
        (),
        ({"status": "failed", "content": {"markdown": "Unusable text"}},),
        ({"status": "completed", "content": {}},),
        ({"status": "completed", "content": {"markdown": " \n"}},),
        ({"status": "completed", "content": {"markdown": "A"}}, {"status": "completed", "content": {"markdown": "B"}}),
    ],
)
async def test_http_parser_rejects_missing_or_failed_markdown(files: tuple[Mapping[str, object], ...]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "task-123"})
        if request.url.path.endswith("/result"):
            return httpx.Response(200, json={"status": "partial", "files": files})
        return httpx.Response(200, json={"status": "partial"})

    parser: Final = HttpMinerUDocumentParser(base_url="https://fixture.invalid", transport=httpx.MockTransport(handler))

    assert await parser.parse("document.pdf", b"pdf", "application/pdf") == ParseFailure(
        ParseFailureKind.UPSTREAM_FAILURE
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [409, 500])
async def test_http_parser_rejects_result_http_failure(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "task-123"})
        if request.url.path.endswith("/result"):
            return httpx.Response(status_code, json={"error": "internal details"})
        return httpx.Response(200, json={"status": "completed"})

    parser: Final = HttpMinerUDocumentParser(base_url="https://fixture.invalid", transport=httpx.MockTransport(handler))

    assert await parser.parse("document.pdf", b"pdf", "application/pdf") == ParseFailure(
        ParseFailureKind.UPSTREAM_FAILURE
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["upload", "poll", "result"])
async def test_http_parser_enforces_total_timeout(monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and phase != "upload":
            monkeypatch.setattr(time, "monotonic", lambda: 104.0)
            return httpx.Response(202, json={"task_id": "task-123"})
        if request.method == "GET" and not request.url.path.endswith("/result"):
            monkeypatch.setattr(time, "monotonic", lambda: 108.0 if phase == "result" else 111.0)
            return httpx.Response(200, json={"status": "completed" if phase == "result" else "queued"})
        monkeypatch.setattr(time, "monotonic", lambda: 111.0)
        await asyncio.Future[None]()
        pytest.fail("The total deadline must cancel the request")

    parser: Final = HttpMinerUDocumentParser(
        base_url="https://fixture.invalid",
        timeout_seconds=10,
        poll_interval_seconds=60,
        transport=httpx.MockTransport(handler),
    )

    assert await parser.parse("document.pdf", b"pdf", "application/pdf") == ParseFailure(ParseFailureKind.TIMEOUT)


@pytest.mark.asyncio
async def test_tool_result_pdf_reaches_chat_as_text_without_changing_other_content() -> None:
    parser: Final = RecordingParser("Extracted document")
    interceptor: Final = MinerUDocumentInterceptor(parser=parser)
    pdf: Final = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(b"pdf-content").decode(),
        },
    }
    image: Final = {"type": "image", "source": {"type": "url", "url": "https://fixture.invalid/image.png"}}
    before: Final = {"type": "text", "text": "Before"}
    after: Final = {"type": "text", "text": "After"}
    tool_result: Final = {
        "type": "tool_result",
        "tool_use_id": "call_document",
        "is_error": False,
        "content": [before, pdf, image, after],
    }
    request: Final = {"messages": [{"role": "user", "content": [tool_result]}]}
    extracted: Final = {"type": "text", "text": "<document>\nExtracted document\n</document>"}
    expected_messages: Final = [
        {
            "role": "user",
            "content": [
                {**tool_result, "content": [before, extracted, image, after]},
            ],
        }
    ]

    result: Final = await interceptor.async_pre_call_hook(None, None, request, "anthropic_messages")

    assert result == {"messages": expected_messages}
    assert request == {"messages": [{"role": "user", "content": [tool_result]}]}
    assert tool_result["content"] == [before, pdf, image, after]
    assert parser.documents == [b"pdf-content"]
    assert LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(result["messages"]) == [
        {
            "role": "tool",
            "tool_call_id": "call_document",
            "content": [
                before,
                extracted,
                {"type": "image_url", "image_url": {"url": "https://fixture.invalid/image.png"}},
                after,
            ],
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("encoded, limit, status", [("invalid!", 100, 400), ("cGRm", 1, 413)])
async def test_tool_result_pdf_obeys_document_validation(encoded: str, limit: int, status: int) -> None:
    parser: Final = RecordingParser("unused")
    interceptor: Final = MinerUDocumentInterceptor(parser=parser, max_file_bytes=limit)
    request: Final = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_document",
                        "content": [
                            {
                                "type": "document",
                                "source": {"type": "base64", "media_type": "application/pdf", "data": encoded},
                            }
                        ],
                    }
                ],
            }
        ]
    }

    with pytest.raises(DocumentInterceptionError) as error:
        await interceptor.async_pre_call_hook(None, None, request, "anthropic_messages")

    assert error.value.status_code == status
    assert parser.documents == []
