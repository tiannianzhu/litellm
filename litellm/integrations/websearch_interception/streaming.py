import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator, Mapping
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME
from litellm.cost_calculator import BaseTokenUsageProcessor
from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger
from litellm.integrations.websearch_interception.transformation import WebSearchTransformation
from litellm.litellm_core_utils.agentic_followup_kwargs import build_agentic_followup_kwargs
from litellm.litellm_core_utils.chat_completion_agentic_loop import (
    agentic_loop_settings,
    check_agentic_loop_safety,
    filter_followup_kwargs,
    with_agentic_loop_metadata,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)
from litellm.utils import CustomStreamWrapper

_MESSAGES: Final = TypeAdapter(list[AllMessageValues])
_REQUEST: Final = TypeAdapter(dict[str, object])


class _SearchOffset(TypedDict):
    websearch_tool_offset: ReadOnly[int]


class _SearchAction(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["search"] = "search"
    query: str


class _SearchResultFields(BaseModel):
    model_config = ConfigDict(frozen=True)
    search_result: str


class _CompletedSearchCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    type: Literal["web_search_call"] = "web_search_call"
    status: Literal["completed"] = "completed"
    action: _SearchAction
    provider_specific_fields: _SearchResultFields


class _SearchMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    websearch_tool_calls: tuple[ChatCompletionDeltaToolCall, ...]
    web_search_calls: tuple[_CompletedSearchCall, ...]
    websearch_native_blocks: tuple[Mapping[str, object], ...]


class SearchArguments(BaseModel):
    query: str
    objective: str | None = None
    search_queries: tuple[str, ...] | None = None


class WebSearchStream(CustomStreamWrapper):
    def __init__(
        self,
        source: CustomStreamWrapper,
        callback: WebSearchInterceptionLogger,
        request: Mapping[str, object],
    ) -> None:
        self._model_name = str(request["model"]).removeprefix("hosted_vllm/")
        super().__init__(
            completion_stream=None,
            model=self._model_name,
            logging_obj=source.logging_obj,
            custom_llm_provider=source.custom_llm_provider,
        )
        self._hidden_params = source._hidden_params
        self._source = source
        self._callback = callback
        self._request = request
        self._iterator = self._iterate()

    async def __anext__(self) -> ModelResponseStream:
        return await self._iterator.__anext__()

    async def aclose(self) -> None:
        try:
            await self._iterator.aclose()
        finally:
            await self._source.aclose()

    async def _iterate(self) -> AsyncGenerator[ModelResponseStream, None]:
        messages: Final = _MESSAGES.validate_python(self._request.get("messages", ()))
        self._messages = cast(list[AllMessageValues], _MESSAGES.dump_python(messages, mode="json"))
        self._chunks = deque[ModelResponseStream]()
        self._usage: tuple[Usage, ...] = ()
        self._finished = False
        self._tool_offset = 0
        self._search_history: tuple[_CompletedSearchCall, ...] = ()
        self._response_id: str | None = None
        self._depth, self._limit, fingerprints = agentic_loop_settings(
            MappingProxyType(
                {
                    **self._request,
                    "max_agentic_loops": self._request.get("max_agentic_loops", self._callback.max_agentic_loops),
                }
            )
        )
        self._fingerprints: tuple[str, ...] = tuple(fingerprints)
        try:
            while not self._finished:
                async for chunk in self._round():
                    yield chunk
        finally:
            await self._source.aclose()

    def _visible(self, chunk: ModelResponseStream) -> ModelResponseStream:
        choices: Final = [
            choice.model_copy(
                update=MappingProxyType(
                    {
                        "finish_reason": None,
                        "delta": choice.delta.model_copy(update=MappingProxyType({"tool_calls": None})),
                    }
                )
            )
            for choice in chunk.choices
        ]
        return chunk.model_copy(update=MappingProxyType({"id": self._response_id, "choices": choices, "usage": None}))

    def _terminal(
        self, calls: tuple[ChatCompletionMessageToolCall, ...], reason: str | None
    ) -> Iterator[ModelResponseStream]:
        if calls:
            offset: Final[_SearchOffset] = {"websearch_tool_offset": self._tool_offset}
            yield ModelResponseStream(
                id=self._response_id,
                model=self._model_name,
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            tool_calls=tuple(
                                ChatCompletionDeltaToolCall(
                                    index=index,
                                    id=call.id,
                                    type=call.type,
                                    function=call.function,
                                )
                                for index, call in enumerate(calls)
                            ),
                            provider_specific_fields=offset,
                        ),
                        finish_reason=None,
                    )
                ],
            )
        yield ModelResponseStream(
            id=self._response_id,
            model=self._model_name,
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(),
                    finish_reason=reason,
                )
            ],
        )
        yield ModelResponseStream(
            id=self._response_id,
            model=self._model_name,
            choices=[],
            usage=BaseTokenUsageProcessor.combine_usage_objects(list(self._usage)),
        )
        self._finished = True

    async def _round(self) -> AsyncGenerator[ModelResponseStream, None]:
        from litellm.main import acompletion, stream_chunk_builder

        self._chunks.clear()
        async for chunk in self._source:
            self._response_id = self._response_id or chunk.id
            self._chunks.append(chunk)
            if any(
                choice.delta.content or getattr(choice.delta, "reasoning_content", None) for choice in chunk.choices
            ):
                yield self._visible(chunk)
        complete: Final = stream_chunk_builder(list(self._chunks))
        if not isinstance(complete, ModelResponse):
            raise ValueError("Search stream ended without a response")
        if len(complete.choices) != 1:
            raise ValueError("Streaming search requires exactly one completion choice")
        choice: Final = complete.choices[0]
        usage: Final = getattr(complete, "usage", None)
        if isinstance(usage, Usage):
            self._usage = (*self._usage, usage)
        calls: Final = tuple(
            call for call in (choice.message.tool_calls or ()) if isinstance(call, ChatCompletionMessageToolCall)
        )
        searches: Final = tuple(call for call in calls if call.function.name == LITELLM_WEB_SEARCH_TOOL_NAME)
        pending: Final = tuple(call for call in calls if call.function.name != LITELLM_WEB_SEARCH_TOOL_NAME)
        if not searches or choice.finish_reason in ("length", "content_filter"):
            for terminal in self._terminal(pending, choice.finish_reason):
                yield terminal
            return
        fingerprint: Final = check_agentic_loop_safety(
            tool_calls=[call.function.model_dump() for call in searches],
            fingerprints=self._fingerprints,
            depth=self._depth,
            max_loops=self._limit,
            model=self._model_name,
        )
        arguments: Final = tuple(SearchArguments.model_validate_json(call.function.arguments) for call in searches)
        results: Final = await asyncio.gather(
            *(
                self._callback._execute_search(  # pyright: ignore[reportPrivateUsage]  # shared search contract
                    argument.query,
                    kwargs=MappingProxyType(self._request),
                    rich=self._callback._rich_search_input(  # pyright: ignore[reportPrivateUsage]  # shared handler
                        argument.model_dump(exclude_none=True)
                    ),
                )
                for argument in arguments
            ),
            return_exceptions=True,
        )
        outcomes: Final = tuple(WebSearchTransformation.search_outcome(result) for result in results)
        outcome_texts: Final = tuple(WebSearchTransformation.search_outcome_text(outcome) for outcome in outcomes)
        self._messages = cast(  # cast-ok: this list adapter dumps JSON messages as a concrete list
            list[AllMessageValues],
            _MESSAGES.dump_python(
                (
                    *self._messages,
                    choice.message.model_dump(exclude_none=True),
                    *(
                        _REQUEST.validate_python(
                            MappingProxyType(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.id,
                                    "content": text,
                                }
                            )
                        )
                        for call, text in zip(searches, outcome_texts, strict=True)
                    ),
                ),
                mode="json",
            ),
        )
        native_pairs: Final = tuple(
            self._callback.native_result_pair(argument.query, outcome)
            for argument, outcome in zip(arguments, outcomes, strict=True)
        )
        self._search_history = (
            *self._search_history,
            *(
                _CompletedSearchCall(
                    id=f"ws_search_{self._depth}_{call.id}",
                    action=_SearchAction(query=argument.query),
                    provider_specific_fields=_SearchResultFields(search_result=text),
                )
                for call, argument, text in zip(searches, arguments, outcome_texts, strict=True)
            ),
        )
        search_metadata: Final = _SearchMetadata(
            websearch_tool_calls=tuple(
                ChatCompletionDeltaToolCall(
                    index=self._tool_offset + index,
                    id=f"search_{self._depth}_{call.id}",
                    type=call.type,
                    function=call.function,
                )
                for index, call in enumerate(searches)
            ),
            web_search_calls=self._search_history,
            websearch_native_blocks=tuple(chain.from_iterable(native_pairs)),
        )
        yield ModelResponseStream(
            id=self._response_id,
            model=self._model_name,
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(provider_specific_fields=search_metadata.model_dump(mode="json")),
                    finish_reason=None,
                )
            ],
        )
        self._tool_offset += len(searches)
        if pending:
            for terminal in self._terminal(pending, "tool_calls"):
                yield terminal
            return
        followup: Final = with_agentic_loop_metadata(
            build_agentic_followup_kwargs(
                request_kwargs=filter_followup_kwargs(_REQUEST.validate_python(self._request)),
                patch_kwargs=MappingProxyType({}),
                request_params=frozenset(("model", "messages", "stream", "tool_choice")),
                depth=self._depth,
                max_loops=self._limit,
                fingerprints=self._fingerprints,
                fingerprint=fingerprint,
            )
        )
        complete_stream: Final = cast(Callable[..., Awaitable[object]], acompletion)
        next_stream: Final = await complete_stream(
            model=f"hosted_vllm/{self._model_name}",
            messages=self._messages,
            stream=True,
            **followup,
        )
        if not isinstance(next_stream, CustomStreamWrapper):
            raise ValueError("Search follow-up did not return a stream")
        self._source = next_stream
        self._depth += 1
        self._fingerprints = (*self._fingerprints, fingerprint)


def streaming_search_callback(
    request: Mapping[str, object],
    provider: str | None,
) -> WebSearchInterceptionLogger | None:
    import litellm
    from litellm.integrations.websearch_interception.tools import is_web_search_tool_chat_completion

    if provider != "hosted_vllm":
        return None
    tools: Final = request.get("tools")
    if not isinstance(tools, list) or not any(is_web_search_tool_chat_completion(tool) for tool in tools):
        return None
    return next(
        (
            callback
            for callback in litellm.callbacks
            if isinstance(callback, WebSearchInterceptionLogger) and provider in callback.enabled_providers
        ),
        None,
    )


def wrap_websearch_stream(
    response: CustomStreamWrapper,
    request: Mapping[str, object],
    provider: str | None,
) -> CustomStreamWrapper:
    if request.get("_agentic_loop_depth"):
        return response
    callback: Final = streaming_search_callback(request, provider)
    return WebSearchStream(response, callback, request) if callback is not None else response
