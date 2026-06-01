#!/usr/bin/env python3
import json
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK_URL", "").strip()
PORT = int(os.environ.get("PORT", "5001"))


def _annotation(alert, name, default=""):
    return alert.get("annotations", {}).get(name, default)


def _label(labels, name):
    value = labels.get(name, "")
    return value if value else None


def _target(labels):
    return (
        _label(labels, "service")
        or _label(labels, "litellm_model_name")
        or _label(labels, "host")
        or _label(labels, "instance")
        or "unknown"
    )


def _details(labels):
    fields = (
        ("model", _label(labels, "litellm_model_name")),
        ("model_id", _label(labels, "model_id")),
        ("provider", _label(labels, "api_provider")),
        ("api_base", _label(labels, "api_base")),
        ("host", _label(labels, "host")),
        ("instance", _label(labels, "instance")),
    )
    return ", ".join(f"{name}: {value}" for name, value in fields if value)


def _status_title(status):
    if status == "firing":
        return "告警触发"
    if status == "resolved":
        return "告警恢复"
    return f"告警状态: {status.upper()}"


def build_text(payload):
    status = payload.get("status", "unknown").lower()
    alerts = payload.get("alerts", [])
    lines = [f"LiteLLM/模型服务{_status_title(status)}", f"数量: {len(alerts)}"]
    for i, alert in enumerate(alerts[:8], start=1):
        labels = alert.get("labels", {})
        name = labels.get("alertname", "unknown")
        severity = labels.get("severity", "unknown")
        summary = _annotation(alert, "summary")
        desc = _annotation(alert, "description")
        details = _details(labels)
        lines.extend(["", f"{i}. [{severity}] {name}", f"对象: {_target(labels)}"])
        if details:
            lines.append(f"详情: {details}")
        if summary:
            lines.append(f"摘要: {summary}")
        if desc:
            lines.append(f"说明: {desc}")
    if len(alerts) > 8:
        lines.append(f"\n其余 {len(alerts) - 8} 条已省略，请看 Prometheus / Alertmanager。")
    return "\n".join(lines)


class Handler(BaseHTTPRequestHandler):
    def _respond(self, code, body=b""):
        self.send_response(code)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, b"ok")
            return
        self._respond(404)

    def do_POST(self):
        if self.path != "/alert":
            self._respond(404)
            return
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._respond(400, str(exc).encode())
            return
        if not WEBHOOK_URL:
            self._respond(500, b"FEISHU_WEBHOOK_URL is not set")
            return
        body = json.dumps(
            {"msg_type": "text", "content": {"text": build_text(payload)}},
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            WEBHOOK_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                self._respond(200 if resp.status < 400 else 502, resp.read())
        except Exception as exc:
            self._respond(502, str(exc).encode())

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)


if __name__ == "__main__":
    print(
        f"feishu alert adapter listening on :{PORT}, webhook_configured={bool(WEBHOOK_URL)}",
        flush=True,
    )
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
