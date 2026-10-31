import runpy
import urllib.request
from collections.abc import Mapping
from email.message import Message
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from typing import Final
from unittest.mock import patch
from urllib.response import addinfourl

import pytest

ROOT: Final = Path(__file__).resolve().parents[2]


def _build_feishu_text(status: str) -> str:
    namespace: Final[Mapping[str, object]] = dict(
        runpy.run_path(str(ROOT / "alertmanager-feishu" / "feishu_adapter.py"))
    )
    build_text: Final = namespace.get("build_text")
    assert callable(build_text)
    result: Final = build_text({"status": status, "alerts": []})
    assert isinstance(result, str)
    return result


def test_feishu_text_distinguishes_firing_and_resolved() -> None:
    assert _build_feishu_text("firing").startswith("LiteLLM/模型服务告警触发")
    assert _build_feishu_text("resolved").startswith("LiteLLM/模型服务告警恢复")


@pytest.mark.parametrize(
    ("upstream_body", "expected_status"),
    (
        pytest.param(b'{"code": 0}', 200, id="delivered"),
        pytest.param(b'{"code": 1}', 502, id="business-error"),
        pytest.param(b"not-json", 502, id="malformed-json"),
        pytest.param(b"{}", 502, id="missing-code"),
        pytest.param(b'{"code": false}', 502, id="invalid-code"),
    ),
)
def test_feishu_handler_acknowledges_only_delivered_alerts(
    monkeypatch: pytest.MonkeyPatch, upstream_body: bytes, expected_status: int
) -> None:
    monkeypatch.setenv("FEISHU_WEBHOOK_URL", "https://example.invalid/feishu")
    namespace: Final[Mapping[str, object]] = dict(
        runpy.run_path(str(ROOT / "alertmanager-feishu" / "feishu_adapter.py"))
    )
    handler_class: Final = namespace["Handler"]
    assert isinstance(handler_class, type) and issubclass(handler_class, BaseHTTPRequestHandler)
    handler: Final = handler_class.__new__(handler_class)
    request_body: Final = b'{"status":"firing","alerts":[]}'
    handler.rfile = BytesIO(
        b"POST /alert HTTP/1.0\r\nHost: localhost\r\nContent-Length: "
        + str(len(request_body)).encode()
        + b"\r\n\r\n"
        + request_body
    )
    response_buffer: Final = BytesIO()
    handler.wfile = response_buffer
    handler.client_address = ("127.0.0.1", 1)
    upstream: Final = addinfourl(BytesIO(upstream_body), Message(), "https://example.invalid/feishu", code=200)

    with patch.object(urllib.request, "urlopen", return_value=upstream) as send:
        handler.handle_one_request()

    send.assert_called_once()
    status_line: Final = response_buffer.getvalue().split(b"\r\n", 1)[0]
    assert int(status_line.split()[1]) == expected_status
