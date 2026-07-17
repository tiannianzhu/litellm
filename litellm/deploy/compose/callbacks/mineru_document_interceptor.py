import asyncio
import base64
import binascii
import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from html import escape
from typing import Annotated, Final, Literal, Protocol, TypeAlias

import anyio
import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.clientside_credential_handler import (
    clientside_credential_keys,
    is_clientside_credential,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.routing_common_utils import provider_for_generic_call
from litellm.llms.anthropic.messages.native.handler import _native_messages_config
from litellm.types.router import DeploymentTypedDict
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


ParseResult: TypeAlias = ParseSuccess | ParseFailure
MinerUTier: TypeAlias = Literal["flash", "basic", "standard", "advanced"]
MinerUOcrMode: TypeAlias = Literal["auto", "txt", "ocr"]
_DEFAULT_MAX_EXTRACTED_CHARACTERS: Final = 300_000
_TextLimit: TypeAlias = Annotated[int, Field(gt=0)]
_TEXT_LIMIT: Final[TypeAdapter[int]] = TypeAdapter(_TextLimit)


class DocumentParser(Protocol):
    async def parse(self, filename: str, content: bytes, media_type: str) -> ParseResult: ...


class _TaskSubmission(BaseModel, frozen=True):
    task_id: str = Field(min_length=1)


class _TaskStatus(BaseModel, frozen=True):
    status: Literal["queued", "running", "completed", "partial", "failed", "canceled"]


class _DocumentContent(BaseModel, frozen=True):
    markdown: str


class _ParsedDocument(BaseModel, frozen=True):
    status: Literal["completed"]
    content: _DocumentContent


class _TaskResult(BaseModel, frozen=True):
    status: Literal["completed", "partial"]
    files: tuple[_ParsedDocument, ...]


class MinerUDocumentInterceptionSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = Field(min_length=1)
    timeout_seconds: float = Field(default=90, gt=0)
    poll_interval_seconds: float = Field(default=1, gt=0)
    max_file_size_mb: int = Field(default=20, gt=0)
    max_extracted_characters: _TextLimit = _DEFAULT_MAX_EXTRACTED_CHARACTERS
    tier: MinerUTier = "standard"
    ocr_mode: MinerUOcrMode = "auto"


class HttpMinerUDocumentParser:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 90,
        poll_interval_seconds: float = 1,
        tier: MinerUTier = "standard",
        ocr_mode: MinerUOcrMode = "auto",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._tier = tier
        self._ocr_mode = ocr_mode
        self._transport = transport

    async def parse(self, filename: str, content: bytes, media_type: str) -> ParseResult:
        try:
            with anyio.fail_after(self._timeout_seconds):
                async with httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=self._timeout_seconds,
                    transport=self._transport,
                    trust_env=False,
                ) as client:
                    submission: Final = await client.post(
                        "v1/tasks",
                        files=(("files", (filename, content, media_type)),),
                        params={
                            "tier": self._tier,
                            "ocr_mode": self._ocr_mode,
                            "page_range": "all",
                            "output_formats": "markdown",
                        },
                    )
                    if submission.status_code != 202:
                        return ParseFailure(ParseFailureKind.UPSTREAM_FAILURE)
                    submission_data: Final = _TaskSubmission.model_validate_json(submission.content)
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
            status_response = await client.get(f"v1/tasks/{task_id}")
            if status_response.status_code != 200:
                return ParseFailure(ParseFailureKind.UPSTREAM_FAILURE)
            status = _TaskStatus.model_validate_json(status_response.content).status
            match status:
                case "failed" | "canceled":
                    return ParseFailure(ParseFailureKind.UPSTREAM_FAILURE)
                case "completed" | "partial":
                    if (result := await self._fetch_result(client, task_id)) is not None:
                        return result
                case "queued" | "running":
                    pass
            await asyncio.sleep(self._poll_interval_seconds)

    async def _fetch_result(self, client: httpx.AsyncClient, task_id: str) -> ParseResult | None:
        response: Final = await client.get(f"v1/tasks/{task_id}/result")
        if response.status_code == 202:
            return None
        if response.status_code != 200:
            return ParseFailure(ParseFailureKind.UPSTREAM_FAILURE)
        result: Final = _TaskResult.model_validate_json(response.content)
        if len(result.files) != 1 or not result.files[0].content.markdown.strip():
            return ParseFailure(ParseFailureKind.UPSTREAM_FAILURE)
        return ParseSuccess(markdown=result.files[0].content.markdown)


class DocumentInterceptionError(Exception):
    def __init__(self, status_code: int, public_message: str) -> None:
        super().__init__(public_message)
        self.status_code = status_code
        self.code = status_code
        self.message = public_message
        self.public_message = public_message
        self.type = "document_interception_error"


def _string_mapping(value: object) -> Mapping[str, object] | None:
    mapping: Final = as_str_mapping(value)
    return mapping if isinstance(value, Mapping) and all(isinstance(key, str) for key in value) else None


def _proxy_router_for_native_documents() -> litellm.Router | None:
    from litellm.proxy.hooks.sensitive_data_routing import PROXY_SensitiveDataRoutingHandler
    from litellm.proxy.proxy_server import llm_router
    from litellm.proxy.utils import ProxyLogging

    capabilities: Final = ProxyLogging.callback_capabilities()
    if capabilities.has_guardrail or any(
        isinstance(callback, PROXY_SensitiveDataRoutingHandler) for callback in capabilities.resolved_callbacks
    ):
        return None
    return llm_router


def _supports_native_documents(deployment: DeploymentTypedDict) -> bool:
    info: Final = _string_mapping(deployment.get("model_info"))
    if info is None or info.get("supports_pdf_input") is not True:
        return False
    params: Final = deployment["litellm_params"]
    try:
        model, provider, _, api_base = litellm.get_llm_provider(
            model=params["model"],
            custom_llm_provider=provider_for_generic_call(params),
            api_base=params.get("api_base"),
        )
    except litellm.BadRequestError:
        return False
    return _native_messages_config(model, provider, info, api_base) is not None


def _native_document_route(router: litellm.Router, data: Mapping[str, object]) -> bool:
    model: Final = data.get("model")
    if not isinstance(model, str) or data.get("guardrails") or is_clientside_credential(data):
        return False
    settings: Final = router.settings()
    raw_defaults: Final[object] = settings.default_litellm_params
    defaults: Final = _string_mapping(raw_defaults)
    if any(
        data.get(key) is not None or (defaults is not None and defaults.get(key) is not None)
        for key in (
            *clientside_credential_keys,
            "custom_llm_provider",
            "specific_deployment",
            "model_id",
            "passthrough_on_no_deployment",
            "user_config",
            "router_settings_override",
            "fallbacks",
            "context_window_fallbacks",
            "content_policy_fallbacks",
        )
    ):
        return False
    if (
        settings.default_deployment is not None
        or settings.fallbacks
        or settings.context_window_fallbacks
        or settings.content_policy_fallbacks
    ):
        return False

    registry: Final = router.registry()
    if settings.router_general_settings.pass_through_all_models and model not in registry.names(kind="group_set"):
        return False
    deployments: Final = registry.deployments(model, source="routable", request=data, call_type="anthropic_messages")
    return bool(deployments) and all(_supports_native_documents(deployment) for deployment in deployments)


class MinerUDocumentInterceptor(CustomLogger):
    def __init__(
        self,
        parser: DocumentParser | None = None,
        max_file_bytes: int = 20 * 1024 * 1024,
        max_extracted_characters: int = _DEFAULT_MAX_EXTRACTED_CHARACTERS,
        router_provider: Callable[[], litellm.Router | None] | None = None,
    ) -> None:
        self._parser = parser
        self._max_file_bytes = max_file_bytes
        self._max_extracted_characters = max_extracted_characters
        self._router_provider = router_provider

    async def async_pre_call_hook(
        self,
        user_api_key_dict: object,
        cache: object,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:
        if call_type != "anthropic_messages":
            return data
        messages: Final = data.get("messages")
        if not isinstance(messages, list):
            return data

        @functools.cache
        def native_route() -> bool:
            router: Final = self._router_provider() if self._router_provider is not None else None
            return router is not None and _native_document_route(router, data)

        transformed_messages: Final = [
            await self._transform_content(fields, native_route)
            if (fields := _string_mapping(message)) is not None
            else message
            for message in messages
        ]
        if all(before is after for before, after in zip(messages, transformed_messages, strict=True)):
            return data
        return {**data, "messages": transformed_messages}

    def _configure(self) -> None:
        if self._parser is not None:
            return
        raw_settings: Final[object] = getattr(litellm, "mineru_document_interception", None)
        try:
            settings: Final = MinerUDocumentInterceptionSettings.model_validate(raw_settings)
        except ValidationError:
            raise DocumentInterceptionError(500, "MinerU document interception is not configured correctly") from None
        self._parser = HttpMinerUDocumentParser(
            base_url=settings.base_url,
            timeout_seconds=settings.timeout_seconds,
            poll_interval_seconds=settings.poll_interval_seconds,
            tier=settings.tier,
            ocr_mode=settings.ocr_mode,
        )
        self._max_file_bytes = settings.max_file_size_mb * 1024 * 1024
        self._max_extracted_characters = settings.max_extracted_characters

    def _configured_text_limit(self) -> int:
        if self._parser is not None:
            return self._max_extracted_characters
        settings: Final = _string_mapping(getattr(litellm, "mineru_document_interception", None))
        if settings is None or "max_extracted_characters" not in settings:
            return self._max_extracted_characters
        try:
            return _TEXT_LIMIT.validate_python(settings["max_extracted_characters"])
        except ValidationError:
            raise DocumentInterceptionError(500, "MinerU document interception is not configured correctly") from None

    async def _transform_content(
        self, message: Mapping[str, object], native_route: Callable[[], bool]
    ) -> Mapping[str, object]:
        content: Final = message.get("content")
        if not isinstance(content, list):
            return message
        transformed_content: Final = [await self._transform_block(block, native_route) for block in content]
        if all(before is after for before, after in zip(content, transformed_content, strict=True)):
            return message
        return {**message, "content": transformed_content}

    async def _transform_block(self, block: object, native_route: Callable[[], bool]) -> object:
        fields: Final = _string_mapping(block)
        if fields is None:
            return block
        if fields.get("type") == "tool_result":
            return await self._transform_content(fields, native_route)
        if fields.get("type") != "document" or native_route():
            return block
        source: Final = _string_mapping(fields.get("source"))
        if source is None:
            raise DocumentInterceptionError(400, "Document block is missing a valid source")
        source_type: Final = source.get("type")
        if source_type == "text":
            text: Final = source.get("data")
            if not isinstance(text, str):
                raise DocumentInterceptionError(400, "Text document source is missing text data")
            self._validate_extracted_text(text, self._configured_text_limit())
            return self._text_block(fields, text)
        if source_type != "base64":
            raise DocumentInterceptionError(400, "Only text and base64 document sources are supported")
        if source.get("media_type") != "application/pdf":
            raise DocumentInterceptionError(415, "Only PDF document uploads are currently supported")
        self._configure()
        content: Final = self._decode_pdf(source.get("data"))
        assert self._parser is not None
        result: Final = await self._parser.parse("document.pdf", content, "application/pdf")
        if isinstance(result, ParseSuccess):
            self._validate_extracted_text(result.markdown, self._max_extracted_characters)
            return self._text_block(fields, result.markdown)
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

    @staticmethod
    def _validate_extracted_text(text: str, limit: int) -> None:
        if not text.strip():
            raise DocumentInterceptionError(400, "Document contains no extractable text")
        if len(text) > limit:
            raise DocumentInterceptionError(413, "Extracted document text is too large")

    @staticmethod
    def _text_block(block: Mapping[str, object], text: str) -> Mapping[str, object]:
        title: Final = block.get("title")
        context: Final = block.get("context")
        title_attribute: Final = (
            f' title="{escape(title[:512], quote=True)}"' if isinstance(title, str) and title else ""
        )
        context_line: Final = (
            f"<context>{escape(context[:2_000])}</context>\n" if isinstance(context, str) and context else ""
        )
        text_block: Final[Mapping[str, object]] = {
            "type": "text",
            "text": f"<document{title_attribute}>\n{context_line}{text}\n</document>",
        }
        cache_control: Final = as_str_mapping(block.get("cache_control"))
        if cache_control is not None:
            return {**text_block, "cache_control": dict(cache_control)}
        return text_block


proxy_handler_instance = MinerUDocumentInterceptor(router_provider=_proxy_router_for_native_documents)
