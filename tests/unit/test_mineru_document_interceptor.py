import asyncio
import base64
from collections.abc import Sequence
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
from litellm.llms.anthropic.pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter


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
async def test_mineru_timeout_is_returned_as_gateway_timeout() -> None:
    class TimeoutParser:
        async def parse(self, filename: str, content: bytes, media_type: str):
            return ParseFailure(ParseFailureKind.TIMEOUT)

    interceptor = MinerUDocumentInterceptor(parser=TimeoutParser())
    pdf = base64.b64encode(b"pdf-content").decode()

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

    assert exc_info.value.status_code == 504


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
async def test_http_parser_uses_async_mineru_task_workflow() -> None:
    requests: list[tuple[str, str]] = []
    statuses: Sequence[str] = ("processing", "completed")
    status_index = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal status_index
        requests.append((request.method, request.url.path))
        if request.method == "POST":
            assert b'name="backend"' in request.content
            assert b"pipeline" in request.content
            assert b'name="parse_method"' in request.content
            assert b"txt" in request.content
            assert b'name="formula_enable"\r\n\r\ntrue' in request.content
            assert b'name="table_enable"\r\n\r\ntrue' in request.content
            assert b'name="image_analysis"\r\n\r\nfalse' in request.content
            assert b'name="return_images"\r\n\r\nfalse' in request.content
            return httpx.Response(202, json={"task_id": "task-123"})
        if request.url.path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "backend": "hybrid-engine",
                    "version": "3.4.4",
                    "results": {"document": {"md_content": "# Parsed"}},
                },
            )
        status = statuses[status_index]
        status_index += 1
        return httpx.Response(200, json={"task_id": "task-123", "status": status})

    parser = HttpMinerUDocumentParser(
        base_url="https://cmpdc.iphy.ac.cn/mineru",
        timeout_seconds=5,
        poll_interval_seconds=0,
        backend="pipeline",
        parse_method="txt",
        formula_enable=True,
        table_enable=True,
        image_analysis=False,
        transport=httpx.MockTransport(handler),
    )

    result = await parser.parse("document.pdf", b"pdf", "application/pdf")

    assert result == ParseSuccess(markdown="# Parsed")
    assert requests == [
        ("POST", "/mineru/tasks"),
        ("GET", "/mineru/tasks/task-123"),
        ("GET", "/mineru/tasks/task-123"),
        ("GET", "/mineru/tasks/task-123/result"),
    ]


@pytest.mark.asyncio
async def test_http_parser_enforces_total_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(202, json={"task_id": "task-123"})

    parser = HttpMinerUDocumentParser(
        base_url="https://mineru.internal.example",
        timeout_seconds=0.01,
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
