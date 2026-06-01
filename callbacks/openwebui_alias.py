from collections.abc import Mapping
from urllib.parse import unquote

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.repositories.table_repositories import EndUserRepository


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(unquote(value).split())
    return normalized or None


def _header(headers: Mapping[str, object], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == wanted:
            return _text(value)
    return None


def _alias(name: str | None, email: str | None) -> str | None:
    if name is not None and email is not None:
        return f"{name} <{email}>"[:512]
    value = name or email
    return value[:512] if value else None


def _end_user(kwargs: Mapping[str, object]) -> str | None:
    litellm_params = _mapping(kwargs.get("litellm_params"))
    metadata = _mapping(litellm_params.get("metadata"))
    end_user = _text(metadata.get("user_api_key_end_user_id"))
    if end_user is not None:
        return end_user

    proxy_request = _mapping(litellm_params.get("proxy_server_request"))
    body = _mapping(proxy_request.get("body"))
    return _text(body.get("user"))


class OpenWebUIAlias(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            end_user = _end_user(_mapping(kwargs))
            if end_user is None or end_user == "system":
                return

            litellm_params = _mapping(_mapping(kwargs).get("litellm_params"))
            proxy_request = _mapping(litellm_params.get("proxy_server_request"))
            headers = _mapping(proxy_request.get("headers"))
            alias = _alias(
                name=_header(headers, "x-openwebui-user-name"),
                email=_header(headers, "x-openwebui-user-email"),
            )
            if alias is None:
                return

            from litellm.proxy.proxy_server import prisma_client

            if prisma_client is None:
                return

            table = EndUserRepository(prisma_client).table
            existing = await table.find_unique(where={"user_id": end_user})
            if existing is None:
                await table.create(
                    data={
                        "user_id": end_user,
                        "alias": alias,
                        "spend": 0,
                        "blocked": False,
                    }
                )
                return

            if getattr(existing, "alias", None) == alias:
                return

            await table.update(where={"user_id": end_user}, data={"alias": alias})
        except Exception:
            verbose_proxy_logger.exception("Failed to sync OpenWebUI end-user alias")


proxy_handler_instance = OpenWebUIAlias()
