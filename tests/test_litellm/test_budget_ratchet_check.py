"""Tests for scripts/budget_ratchet_check.py.

The guard's contract is "limits may only fall": a raised limit, a dropped rule, or
a deleted file is a regression, while a lowered/equal limit, a brand-new rule, or a
brand-new budget file is fine. Each branch is pinned here.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "budget_ratchet_check.py"
)
_spec = importlib.util.spec_from_file_location("budget_ratchet_check", _MODULE_PATH)
ratchet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ratchet)


def _spec_of(limit):
    return {"limit": limit}


def test_limits_read_the_limit_and_skip_malformed():
    limits = ratchet._limits({"LIT006": _spec_of(1023), "junk": 5})
    assert limits == {"LIT006": 1023}  # malformed (non-dict) spec ignored


def test_limits_fall_back_to_legacy_baseline_plus_slack():
    # The base side of a diff can predate the `limit` migration; its ceiling is
    # baseline + slack, read on the same footing as a new-schema `limit`.
    assert ratchet._limits({"LIT006": {"baseline": 1013, "slack": 10}}) == {"LIT006": 1023}


def test_migration_from_legacy_schema_to_equal_limit_is_clean():
    # baseline+slack (1023) -> limit 1023 is the same ceiling, so no regression.
    base = {"LIT006": {"baseline": 1013, "slack": 10}}
    assert ratchet.regressions_for("b.json", base, {"LIT006": _spec_of(1023)}) == []
    # ...and a genuine raise across the migration is still caught.
    regs = ratchet.regressions_for("b.json", base, {"LIT006": _spec_of(1024)})
    assert [r.rule for r in regs] == ["LIT006"] and "1023 -> 1024" in regs[0].detail


def test_raised_limit_is_a_regression():
    base = {"LIT006": _spec_of(1023)}
    head = {"LIT006": _spec_of(1024)}
    regs = ratchet.regressions_for("b.json", base, head)
    assert [r.rule for r in regs] == ["LIT006"]
    assert "1023 -> 1024" in regs[0].detail


def test_lowered_or_equal_limit_is_clean():
    base = {"LIT006": _spec_of(1023)}
    # limit drops
    assert ratchet.regressions_for("b.json", base, {"LIT006": _spec_of(1000)}) == []
    # nothing changes
    assert ratchet.regressions_for("b.json", base, {"LIT006": _spec_of(1023)}) == []


def test_dropped_rule_is_a_regression():
    regs = ratchet.regressions_for("b.json", {"LIT007": _spec_of(0)}, {})
    assert [r.rule for r in regs] == ["LIT007"]
    assert "dropped" in regs[0].detail


def test_new_rule_in_head_is_clean():
    assert ratchet.regressions_for("b.json", {}, {"new-rule": _spec_of(5)}) == []


def test_dropped_rule_that_graduated_to_a_hard_failing_config_is_clean():
    base = {"UP006": _spec_of(0)}
    assert ratchet.regressions_for("b.json", base, {}, graduated=("UP006",)) == []


def test_graduation_matches_by_prefix_like_ruff_selectors_do():
    base = {"ANN202": _spec_of(865)}
    assert ratchet.regressions_for("b.json", base, {}, graduated=("ANN",)) == []


def test_an_unrelated_graduation_does_not_excuse_a_dropped_rule():
    base = {"C901": _spec_of(3)}
    regs = ratchet.regressions_for("b.json", base, {}, graduated=("UP006", "SIM118"))
    assert [r.rule for r in regs] == ["C901"]
    assert "dropped" in regs[0].detail


def test_graduation_never_excuses_a_raised_limit():
    base = {"UP006": _spec_of(0)}
    regs = ratchet.regressions_for("b.json", base, {"UP006": _spec_of(7)}, graduated=("UP006",))
    assert [r.rule for r in regs] == ["UP006"]
    assert "0 -> 7" in regs[0].detail


def test_graduated_selectors_come_from_the_paired_ruff_config():
    selectors = ratchet.graduated_selectors("ruff-strict-budget.json")
    assert "UP006" in selectors
    assert "ANN" not in selectors


def test_budgets_without_a_paired_config_can_never_graduate():
    assert ratchet.graduated_selectors("type-discipline-budget.json") == ()
    assert ratchet.graduated_selectors("basedpyright-code-budget.json") == ()


def test_a_selector_the_config_also_ignores_does_not_count_as_graduated():
    lint = {"ignore": ["UP006"], "extend-select": ["UP006", "SIM118"]}
    assert ratchet.selectors_hard_failed_by(lint) == ("SIM118",)


def test_selectors_hard_failed_by_reads_a_config_with_no_ignore_list():
    assert ratchet.selectors_hard_failed_by({"extend-select": ["UP006"]}) == ("UP006",)


def test_deleted_budget_file_is_a_regression():
    regs = ratchet.regressions_for("b.json", {"LIT006": _spec_of(1)}, None)
    assert [r.rule for r in regs] == ["*"]
    assert "deleted" in regs[0].detail


def test_new_budget_file_has_nothing_to_ratchet():
    assert ratchet.regressions_for("b.json", None, {"LIT006": _spec_of(1)}) == []


def test_default_budgets_watch_every_budget_file_in_the_repo():
    # This job is the repo's only ceiling-raise alarm, so every *-budget.json on disk must be
    # watched; a budget left out of DEFAULT_BUDGETS (e.g. basedpyright-code-budget.json) can be
    # loosened with no signal. Equality also catches a phantom entry that no longer exists.
    repo_root = _MODULE_PATH.parents[1]
    on_disk = frozenset(p.name for p in repo_root.glob("*budget*.json"))
    assert on_disk == frozenset(ratchet.DEFAULT_BUDGETS)


# --------------------------------------------------------------------------- #
# Base-ref resolution: a bad ref must fail loudly, never pass vacuously
# --------------------------------------------------------------------------- #


def test_ref_is_commit_distinguishes_real_from_bogus():
    assert ratchet._ref_is_commit("HEAD") is True
    assert ratchet._ref_is_commit("definitely-not-a-real-ref-zzz") is False


def test_load_base_reads_a_present_file_and_none_for_an_absent_one():
    # A real budget file exists at HEAD; a made-up path is absent at the same (valid) ref.
    assert ratchet._load_base("type-discipline-budget.json", "HEAD") is not None
    assert ratchet._load_base("scripts/no-such-budget-xyz.json", "HEAD") is None


def test_unresolvable_base_ref_exits_nonzero_instead_of_skipping():
    proc = subprocess.run(
        [sys.executable, str(_MODULE_PATH), "--base", "definitely-not-a-real-ref-zzz"],
        cwd=_MODULE_PATH.parents[1],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "Cannot resolve the merge base with local definitely-not-a-real-ref-zzz" in proc.stderr


def _git_in(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Budget tests",
            "-c",
            "user.email=budget-tests@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        text=True,
    ).strip()


@pytest.fixture(params=("ruff_strict_gate", "type_discipline_gate", "type_check_gate", "test_quality_gate"))
def updater(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    gate_name: Final[str] = request.param
    spec: Final = importlib.util.spec_from_file_location(
        f"budget_update_{gate_name}", _MODULE_PATH.parent / f"{gate_name}.py"
    )
    assert spec is not None and spec.loader is not None
    updater: Final = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, updater)
    spec.loader.exec_module(updater)
    return updater


def test_budget_update_does_not_charge_committed_fixes_twice(
    updater: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git_in(tmp_path, "init", "-q", "-b", "main")
    budget_name: Final = updater.BUDGET_PATH.name
    budget_path: Final = tmp_path / budget_name
    budget_path.write_text(json.dumps({"EXAMPLE": {"limit": 100}}))
    _git_in(tmp_path, "add", budget_name)
    _git_in(tmp_path, "commit", "-qm", "initial budget")
    base: Final = _git_in(tmp_path, "rev-parse", "HEAD")
    _git_in(tmp_path, "checkout", "-qb", "local")
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(updater, "BUDGET_PATH", budget_path)
    monkeypatch.setattr(updater, "resolve_base_point", lambda ref: _git_in(tmp_path, "merge-base", ref, "HEAD"))
    if budget_name == "basedpyright-code-budget.json":
        monkeypatch.setattr(updater, "base_counts_cached", lambda ref: {"EXAMPLE": 100})
    else:
        monkeypatch.setattr(updater, "base_counts", lambda ref: {"EXAMPLE": 100})
        monkeypatch.setattr(updater, "head_violations", lambda: (updater.Violation("sample.py", 1, "EXAMPLE"),) * 95)

    def update() -> None:
        if budget_name == "basedpyright-code-budget.json":
            updater.cmd_update({"EXAMPLE": 95}, base)
        else:
            updater.cmd_update(base)

    update()
    once: Final = budget_path.read_bytes()
    assert json.loads(once) == {"EXAMPLE": {"limit": 95}}
    _git_in(tmp_path, "add", budget_name)
    _git_in(tmp_path, "commit", "-qm", "record fixed violations")
    update()
    assert budget_path.read_bytes() == once


def test_budget_update_counts_additional_fixes_without_loosening_existing_limits(updater: ModuleType) -> None:
    base_budget: Final = {"EXAMPLE": {"limit": 100}, "STRICT": {"limit": 100}, "REMOVED": {"limit": 0}}
    budget: Final = {"EXAMPLE": {"limit": 95}, "STRICT": {"limit": 70}, "NEW": {"limit": 150}}
    base_counts: Final = {"EXAMPLE": 100, "STRICT": 100, "NEW": 1000}
    updated: Final = updater.ratcheted_budget(budget, {"EXAMPLE": 90, "STRICT": 90, "NEW": 0}, base_counts, base_budget)
    assert updated == {"EXAMPLE": {"limit": 90}, "STRICT": {"limit": 70}, "NEW": {"limit": 150}}
    assert (
        updater.ratcheted_budget(updated, {"EXAMPLE": 98, "STRICT": 110, "NEW": 0}, base_counts, base_budget) == updated
    )


@pytest.mark.parametrize(
    ("contents", "expected"),
    (
        (None, {}),
        ('{"EXAMPLE": {"limit": 100}}', {"EXAMPLE": {"limit": 100}}),
        ('{"EXAMPLE": {"baseline": 90, "slack": 10}}', {"EXAMPLE": {"limit": 100}}),
    ),
)
def test_base_budget_reads_the_committed_snapshot(
    updater: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str | None,
    expected: dict[str, dict[str, int]],
) -> None:
    _git_in(tmp_path, "init", "-q", "-b", "main")
    budget_path: Final = tmp_path / updater.BUDGET_PATH.name
    if contents is not None:
        budget_path.write_text(contents)
        _git_in(tmp_path, "add", budget_path.name)
    _git_in(tmp_path, "commit", "-qm", "base snapshot", "--allow-empty")
    base: Final = _git_in(tmp_path, "rev-parse", "HEAD")
    budget_path.write_text('{"EXAMPLE": {"limit": 1}}')
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(updater, "BUDGET_PATH", budget_path)
    assert updater._base_budget(base) == expected


def test_base_budget_does_not_treat_invalid_refs_or_json_as_new_rules(
    updater: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git_in(tmp_path, "init", "-q", "-b", "main")
    budget_path: Final = tmp_path / updater.BUDGET_PATH.name
    budget_path.write_text("invalid JSON")
    _git_in(tmp_path, "add", budget_path.name)
    _git_in(tmp_path, "commit", "-qm", "malformed budget")
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(updater, "BUDGET_PATH", budget_path)
    with pytest.raises(subprocess.CalledProcessError):
        updater._base_budget("missing-ref")
    with pytest.raises(json.JSONDecodeError):
        updater._base_budget("HEAD")
