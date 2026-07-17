import asyncio
import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from html import escape
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import CallTypesLiteral


class ParseFailureKind(str, Enum):
    UPSTREAM_FAILURE = "upstream_failure"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class ParseSuccess:
    markdown: str


@dataclass(frozen=True, slots=True)
class ParseFailure:
    kind: ParseFailureKind


ParseResult = ParseSuccess | ParseFailure
MinerUBackend = Literal["pipeline", "vlm-engine", "hybrid-engine", "vlm-http-client", "hybrid-http-client"]
MinerUEffort = Literal["medium", "high"]
MinerUParseMethod = Literal["auto", "txt", "ocr"]


class DocumentParser(Protocol):
    async def parse(self, filename: str, content: bytes, media_type: str) -> ParseResult: ...


class _TaskSubmission(BaseModel):
    task_id: str


class _TaskStatus(BaseModel):
    status: str


class _ParsedDocument(BaseModel):
    md_content: str


class _TaskResult(BaseModel):
    results: dict[str, _ParsedDocument]


class MinerUDocumentInterceptionSettings(BaseModel):
    base_url: str = Field(min_length=1)
    timeout_seconds: float = Field(default=90, gt=0)
    poll_interval_seconds: float = Field(default=1, gt=0)
    max_file_size_mb: int = Field(default=20, gt=0)
    max_extracted_characters: int = Field(default=300_000, gt=0)
    backend: MinerUBackend = "hybrid-engine"
    effort: MinerUEffort = "medium"
    parse_method: MinerUParseMethod = "auto"
    formula_enable: bool = True
    table_enable: bool = True
    image_analysis: bool = False


class HttpMinerUDocumentParser:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 90,
        poll_interval_seconds: float = 1,
        backend: MinerUBackend = "hybrid-engine",
        effort: MinerUEffort = "medium",
        parse_method: MinerUParseMethod = "auto",
        formula_enable: bool = True,
        table_enable: bool = True,
        image_analysis: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._backend = backend
        self._effort = effort
        self._parse_method = parse_method
        self._formula_enable = formula_enable
        self._table_enable = table_enable
        self._image_analysis = image_analysis
        self._transport = transport

    async def parse(self, filename: str, content: bytes, media_type: str) -> ParseResult:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=self._timeout_seconds,
                    transport=self._transport,
                    trust_env=False,
                ) as client:
                    submission = await client.post(
                        "/tasks",
                        files=[("files", (filename, content, media_type))],
                        data={
                            "backend": self._backend,
                            "effort": self._effort,
                            "parse_method": self._parse_method,
                            "lang_list": "ch",
                            "formula_enable": str(self._formula_enable).lower(),
                            "table_enable": str(self._table_enable).lower(),
                            "image_analysis": str(self._image_analysis).lower(),
                            "return_md": "true",
                            "return_images": "false",
                            "return_content_list": "false",
                        },
                    )
                    if submission.status_code != 202:
                        return ParseFailure(ParseFailureKind.UPSTREAM_FAILURE)
                    submission_data = _TaskSubmission.model_validate_json(submission.content)
                    return await self._wait_for_result(client, submission_data.task_id)
        except (TimeoutError, httpx.TimeoutException):
            return ParseFailure(ParseFailureKind.TIMEOUT)
        except (httpx.HTTPError, ValidationError):
            return ParseFailure(ParseFailureKind.UPSTREAM_FAILURE)

    async def _wait_for_result(
        self,
        client: httpx.AsyncClient,
        task_id: str,
    ) -> ParseResult:
        while True:
            status_response = await client.get(f"/tasks/{task_id}")
            if status_response.status_code != 200:
                return ParseFailure(ParseFailureKind.UPSTREAM_FAILURE)
            status = _TaskStatus.model_validate_json(status_response.content).status
            if status == "failed":
                return ParseFailure(ParseFailureKind.UPSTREAM_FAILURE)
            if status == "completed":
                result = await self._fetch_result(client, task_id)
                if result is not None:
                    return result
            elif status not in {"pending", "processing"}:
                return ParseFailure(ParseFailureKind.UPSTREAM_FAILURE)
            await asyncio.sleep(self._poll_interval_seconds)

    async def _fetch_result(self, client: httpx.AsyncClient, task_id: str) -> ParseResult | None:
        response = await client.get(f"/tasks/{task_id}/result")
        if response.status_code == 202:
            return None
        if response.status_code != 200:
            return ParseFailure(ParseFailureKind.UPSTREAM_FAILURE)
        result = _TaskResult.model_validate_json(response.content)
        documents = tuple(result.results.values())
        if len(documents) != 1 or not documents[0].md_content.strip():
            return ParseFailure(ParseFailureKind.UPSTREAM_FAILURE)
        return ParseSuccess(markdown=documents[0].md_content)


class DocumentInterceptionError(Exception):
    def __init__(self, status_code: int, public_message: str) -> None:
        super().__init__(public_message)
        self.status_code = status_code
        self.code = status_code
        self.message = public_message
        self.public_message = public_message
        self.type = "document_interception_error"


def _string_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        return None
    return {str(key): item for key, item in value.items()}


class MinerUDocumentInterceptor(CustomLogger):
    def __init__(
        self,
        parser: DocumentParser | None = None,
        max_file_bytes: int = 20 * 1024 * 1024,
        max_extracted_characters: int = 300_000,
    ) -> None:
        self._parser = parser
        self._max_file_bytes = max_file_bytes
        self._max_extracted_characters = max_extracted_characters

    async def async_pre_call_hook(
        self,
        user_api_key_dict: object,
        cache: object,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:
        if call_type != "anthropic_messages":
            return data
        messages = data.get("messages")
        if not isinstance(messages, list):
            return data
        transformed_messages = [await self._transform_message(message) for message in messages]
        if transformed_messages == messages:
            return data
        return {**data, "messages": transformed_messages}

    def _configure(self) -> None:
        if self._parser is not None:
            return
        import litellm

        raw_settings = getattr(litellm, "mineru_document_interception", None)
        try:
            settings = MinerUDocumentInterceptionSettings.model_validate(raw_settings)
        except ValidationError:
            raise DocumentInterceptionError(500, "MinerU document interception is not configured correctly") from None
        self._parser = HttpMinerUDocumentParser(
            base_url=settings.base_url,
            timeout_seconds=settings.timeout_seconds,
            poll_interval_seconds=settings.poll_interval_seconds,
            backend=settings.backend,
            effort=settings.effort,
            parse_method=settings.parse_method,
            formula_enable=settings.formula_enable,
            table_enable=settings.table_enable,
            image_analysis=settings.image_analysis,
        )
        self._max_file_bytes = settings.max_file_size_mb * 1024 * 1024
        self._max_extracted_characters = settings.max_extracted_characters

    async def _transform_message(self, value: object) -> object:
        message = _string_mapping(value)
        if message is None:
            return value
        content = message.get("content")
        if not isinstance(content, list):
            return message
        transformed_content = [await self._transform_block(block) for block in content]
        return {**message, "content": transformed_content}

    async def _transform_block(self, value: object) -> object:
        block = _string_mapping(value)
        if block is None or block.get("type") != "document":
            return value
        self._configure()
        source = _string_mapping(block.get("source"))
        if source is None:
            raise DocumentInterceptionError(400, "Document block is missing a valid source")
        source_type = source.get("type")
        if source_type == "text":
            text = source.get("data")
            if not isinstance(text, str):
                raise DocumentInterceptionError(400, "Text document source is missing text data")
            self._validate_extracted_text(text)
            return self._text_block(block, text)
        if source_type != "base64":
            raise DocumentInterceptionError(400, "Only text and base64 document sources are supported")
        media_type = source.get("media_type")
        if media_type != "application/pdf":
            raise DocumentInterceptionError(415, "Only PDF document uploads are currently supported")
        content = self._decode_pdf(source.get("data"))
        assert self._parser is not None
        result = await self._parser.parse("document.pdf", content, "application/pdf")
        if isinstance(result, ParseSuccess):
            self._validate_extracted_text(result.markdown)
            return self._text_block(block, result.markdown)
        if result.kind == ParseFailureKind.TIMEOUT:
            raise DocumentInterceptionError(504, "Document parsing timed out")
        raise DocumentInterceptionError(502, "Document parsing service failed")

    def _decode_pdf(self, value: object) -> bytes:
        if not isinstance(value, str):
            raise DocumentInterceptionError(400, "Base64 document source is missing data")
        maximum_encoded_length = ((self._max_file_bytes + 2) // 3) * 4
        if len(value) > maximum_encoded_length:
            raise DocumentInterceptionError(413, "Document is too large")
        try:
            content = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            raise DocumentInterceptionError(400, "Document contains invalid base64 data") from None
        if len(content) > self._max_file_bytes:
            raise DocumentInterceptionError(413, "Document is too large")
        return content

    def _validate_extracted_text(self, text: str) -> None:
        if not text.strip():
            raise DocumentInterceptionError(400, "Document contains no extractable text")
        if len(text) > self._max_extracted_characters:
            raise DocumentInterceptionError(413, "Extracted document text is too large")

    @staticmethod
    def _text_block(block: dict[str, object], text: str) -> dict[str, object]:
        title = block.get("title")
        context = block.get("context")
        title_attribute = f' title="{escape(title[:512], quote=True)}"' if isinstance(title, str) and title else ""
        context_line = f"<context>{escape(context[:2_000])}</context>\n" if isinstance(context, str) and context else ""
        text_block: dict[str, object] = {
            "type": "text",
            "text": f"<document{title_attribute}>\n{context_line}{text}\n</document>",
        }
        cache_control = block.get("cache_control")
        if isinstance(cache_control, Mapping):
            return {**text_block, "cache_control": dict(cache_control)}
        return text_block


proxy_handler_instance = MinerUDocumentInterceptor()
