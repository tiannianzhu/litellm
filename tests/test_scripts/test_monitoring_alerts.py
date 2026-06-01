import runpy
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]


def _build_feishu_text(status: str) -> str:
    namespace: Final[dict[str, object]] = dict(runpy.run_path(str(ROOT / "alertmanager-feishu" / "feishu_adapter.py")))
    build_text: Final = namespace.get("build_text")
    assert callable(build_text)
    result: Final = build_text({"status": status, "alerts": []})
    assert isinstance(result, str)
    return result


def test_feishu_text_distinguishes_firing_and_resolved() -> None:
    assert _build_feishu_text("firing").startswith("LiteLLM/模型服务告警触发")
    assert _build_feishu_text("resolved").startswith("LiteLLM/模型服务告警恢复")
