from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, islice
from typing import Final, Protocol
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import get_or_create_metadata_bucket
from litellm.litellm_core_utils.prompt_templates.factory import resolve_structured_messages


class SessionIdentityCache(Protocol):
    async def async_batch_get_cache(  # mutable-ok: DualCache's established interface accepts a key list
        self, keys: list[str]
    ) -> Sequence[object] | None: ...

    async def async_set_cache(self, key: str, value: str, *, ttl: int) -> object: ...


@dataclass(frozen=True, slots=True)
class _PrefixState:
    digest: bytes
    has_user_message: bool


_INITIAL_PREFIX_STATE: Final = _PrefixState(digest=b"", has_user_message=False)
_PREFIX_CACHE_NAMESPACE: Final = "complexity_router_prefix_session:v1"
DEFAULT_PREFIX_SESSION_TTL_SECONDS: Final = 3600
_MESSAGES_ADAPTER: Final = TypeAdapter(list[dict[str, object]])
_STRING_OBJECT_MAPPING_ADAPTER: Final = TypeAdapter(dict[str, object])


def _structured_messages(value: object) -> list[dict[str, object]] | None:
    if value is None:
        return None
    try:
        return _MESSAGES_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _metadata_dicts(request_data: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _STRING_OBJECT_MAPPING_ADAPTER.validate_python(metadata)
        for key in ("litellm_metadata", "metadata")
        if isinstance(metadata := request_data.get(key), Mapping)
    )


def _first_metadata_value(metadata_dicts: Sequence[Mapping[str, object]], key: str) -> str | None:
    return next(
        (str(value) for metadata in metadata_dicts if (value := metadata.get(key)) is not None and str(value)),
        None,
    )


def _explicit_session_id(request_data: Mapping[str, object]) -> str | None:
    return _first_metadata_value(_metadata_dicts(request_data), "session_id")


def _caller_scope(request_data: Mapping[str, object]) -> str | None:
    metadata_dicts: Final = _metadata_dicts(request_data)
    key_hash: Final = _first_metadata_value(metadata_dicts, "user_api_key_hash")
    end_user: Final = _first_metadata_value(metadata_dicts, "user_api_key_end_user_id")
    internal_user: Final = _first_metadata_value(metadata_dicts, "user_api_key_user_id")
    requester_ip: Final = _first_metadata_value(metadata_dicts, "requester_ip_address")
    user_agent: Final = _first_metadata_value(metadata_dicts, "user_agent")
    principal: Final = next(
        (
            value
            for value in (
                f"end-user:{end_user}" if end_user is not None else None,
                f"network:{requester_ip or ''}\x00{user_agent or ''}" if requester_ip or user_agent else None,
                f"internal-user:{internal_user}" if internal_user is not None else None,
                f"key:{key_hash}" if key_hash is not None else None,
            )
            if value is not None
        ),
        None,
    )
    if principal is None:
        return None
    scope_material: Final = f"{key_hash or 'no-key'}\x00{principal}".encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(scope_material).hexdigest()


def _advance_prefix(state: _PrefixState, message: Mapping[str, object]) -> _PrefixState:
    payload: Final = json.dumps(
        message,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8", errors="surrogatepass")
    framed_payload: Final = len(payload).to_bytes(8, byteorder="big") + payload
    return _PrefixState(
        digest=hashlib.sha256(state.digest + framed_payload).digest(),
        has_user_message=state.has_user_message or message.get("role") == "user",
    )


def _prefix_cache_keys(
    model_name: str,
    caller_scope: str,
    messages: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    states: Final = islice(accumulate(messages, _advance_prefix, initial=_INITIAL_PREFIX_STATE), 1, None)
    return tuple(
        f"{_PREFIX_CACHE_NAMESPACE}:{model_name}:{caller_scope}:{state.digest.hex()}"
        for state in states
        if state.has_user_message
    )


async def _resolve_prefix_session_id(
    *,
    cache: SessionIdentityCache,
    model_name: str,
    request_data: Mapping[str, object],
    messages: Sequence[Mapping[str, object]] | None,
    ttl_seconds: int,
) -> str | None:
    if (session_id := _explicit_session_id(request_data)) is not None:
        return session_id
    if not messages or (caller_scope := _caller_scope(request_data)) is None:
        return None
    prefix_keys: Final = _prefix_cache_keys(model_name, caller_scope, messages)
    if not prefix_keys:
        return None

    candidate_keys: Final = list(  # mutable-ok: DualCache's batch API requires a list
        reversed(prefix_keys)
    )
    cached_session_ids: Final = await cache.async_batch_get_cache(keys=candidate_keys)
    matched_session_id: Final = (
        next((value for value in cached_session_ids if isinstance(value, str) and value), None)
        if cached_session_ids is not None
        else None
    )
    resolved_session_id: Final = matched_session_id or f"prefix-{uuid4()}"
    await cache.async_set_cache(key=prefix_keys[-1], value=resolved_session_id, ttl=ttl_seconds)
    return resolved_session_id


async def apply_inferred_session_id(
    *,
    request_data: dict[str, object],  # mutable-ok: this boundary stamps the inferred session ID in place
    cache: SessionIdentityCache,
    ttl_seconds: int = DEFAULT_PREFIX_SESSION_TTL_SECONDS,
) -> str | None:
    """Infer and stamp an internal session ID for conversation logging."""
    if _explicit_session_id(request_data) is not None:
        return None
    model_name: Final = request_data.get("model")
    if not isinstance(model_name, str):
        return None
    messages: Final = resolve_structured_messages(
        messages=_structured_messages(request_data.get("messages")),
        request_kwargs=request_data,
    )
    try:
        session_id: Final = await _resolve_prefix_session_id(
            cache=cache,
            model_name=model_name,
            request_data=request_data,
            messages=messages,
            ttl_seconds=ttl_seconds,
        )
    except Exception:
        verbose_proxy_logger.warning("Failed to infer session_id from message prefixes", exc_info=True)
        return None
    if session_id is not None:
        _, metadata = get_or_create_metadata_bucket(request_data)
        metadata["session_id"] = session_id
    return session_id
