"""Tests for scripts/00_verify_math.py (M0/D0 numerical verification of P1-P7).

The script is run in-process through main([...]) on CPU in --quick mode; every run writes to pytest's
tmp_path and to a tmp run registry (never manifests/run_registry.jsonl).
"""
import importlib.util
import json
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "00_verify_math.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_math_00", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vm():
    return _load()


@pytest.fixture(scope="module")
def quick_run(vm, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vm")
    out, reg = tmp / "run", tmp / "registry.jsonl"
    code = vm.main(["--quick", "--figures", "--device", "cpu", "--out", str(out), "--registry", str(reg)])
    return code, out, reg


def test_quick_run_passes_and_writes_artifacts(quick_run):
    code, out, reg = quick_run
    assert code == 0
    res = json.loads((out / "verify_math.json").read_text(encoding="utf-8"))
    assert res["all_passed"] is True
    assert {c["section"] for c in res["checks"]} == {"A", "B", "C", "D"}
    for c in res["checks"]:
        assert set(c) == {"section", "name", "value", "tolerance", "passed", "detail"}
        assert c["passed"] and c["value"] <= c["tolerance"]
    meta = res["meta"]
    assert meta["quick"] is True and meta["N_mc"] == 60_000 and meta["dtype"] == "float64"
    assert "git_commit" in meta and meta["tf32"] == {"tf32_matmul": False, "tf32_cudnn": False}
    md = (out / "verify_math.md").read_text(encoding="utf-8")
    for title in ("## A.", "## B.", "## C.", "## D.", "What this does not show"):
        assert title in md
    assert "FAIL" not in md
    for name in ("phase_diagram.png", "keep_curves.png"):
        png = out / "figures" / name
        assert png.exists() and png.read_bytes()[:4] == b"\x89PNG"
    record = json.loads(reg.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert record["status"] == "ok" and record["script"] == "00_verify_math.py"
    assert (out / "summary.json").exists() and (out / "config.yaml").exists()


def test_required_checks_are_present(quick_run):
    _, out, _ = quick_run
    checks = json.loads((out / "verify_math.json").read_text(encoding="utf-8"))["checks"]
    a_cases = {(c["detail"]["family"], c["detail"]["rule"], c["detail"]["w"])
               for c in checks if c["section"] == "A" and "family" in c["detail"] and "rule" in c["detail"]}
    assert len(a_cases) == 4 * 4  # four spectrum families x equal_snr / variance / r2(1.5) / r2(3)
    assert {c["detail"]["source"] for c in checks if c["section"] == "B"} == {"gaussian", "laplace"}
    hand = [c for c in checks if c["name"].startswith("R=0.8R*: signal keep == 0 /")]
    assert sorted(c["detail"]["w"] for c in hand) == [1.0, 1.5, 2.0, 3.0, 5.0]
    assert sum(c["section"] == "D" for c in checks) == 5


def test_closed_form_keep_matches_hand_values(vm):
    R = 0.8 * 0.5 * vm.THEORY["d_n"] * math.log(vm.THEORY["rho"])
    for w, hand in vm.HAND_KEEP.items():
        assert vm.closed_signal_keep(w, 1.0, R) == pytest.approx(hand, abs=0.01)
    assert vm.R_star_w(1.0, 1.0) == pytest.approx(5.0 * math.log(6.0))


def test_a_failing_check_gives_exit_code_1(vm, tmp_path, monkeypatch):
    def broken(chk, seed):
        chk.add("D", "deliberately failing check", 1.0, 0.0)
    monkeypatch.setattr(vm, "section_d", broken)
    reg = tmp_path / "registry.jsonl"
    code = vm.main(["--quick", "--device", "cpu", "--out", str(tmp_path / "run"), "--registry", str(reg)])
    assert code == 1
    assert json.loads(reg.read_text(encoding="utf-8").strip().splitlines()[-1])["status"] == "checks_failed"
    assert "## Failed checks" in (tmp_path / "run" / "verify_math.md").read_text(encoding="utf-8")
