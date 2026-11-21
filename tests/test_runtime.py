"""Tests for rwvae.runtime (ARCHITECTURE_V2.md Sec. 3.1).

Covers config merge/override semantics, fingerprints, JSON conversion, the Run directory
lifecycle (config.yaml, log.txt, metrics.jsonl, summary.json, registry line) including the
exist/overwrite/resume rules, seeding, device selection, file hashing and git provenance.
Everything is written below tmp_path; every Run gets registry=tmp_path/"registry.jsonl".
"""
import argparse
import dataclasses
import hashlib
import json
import logging
import os
import random
import re
import textwrap
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from rwvae.runtime import (Run, add_common_args, config_fingerprint, env_info, get_logger, git_commit,
                           load_config, parse_config_from_args, save_config, seed_everything,
                           select_device, sha256_file, to_jsonable)

RECORD_KEYS = {"time", "script", "status", "out_dir", "config_fingerprint", "git_commit", "env", "summary"}


@pytest.fixture(autouse=True)
def _restore_package_logger():
    """Run attaches handlers to the 'rwvae' logger; restore it so other test files are unaffected."""
    logger = logging.getLogger("rwvae")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    for h in logger.handlers:
        if h not in handlers:
            h.close()
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def _write(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def _lines(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ----------------------------------------------------------------------------- load_config
def test_load_config_deep_merge(tmp_path):
    a = _write(tmp_path / "a.yaml", """
        model: {depth: 12, heads: 6, opt: {lr: 1.0e-3, betas: [0.9, 0.95]}}
        grid: [11, 22, 33]
        name: base
        mixed: {x: 1}
        keep: 7
    """)
    b = _write(tmp_path / "b.yaml", """
        model: {heads: 8, opt: {betas: [0.9, 0.99]}}
        grid: [44]
        name: {first: x}
        mixed: 3
        extra: {k: v}
    """)
    cfg = load_config([a, b])
    assert cfg == {
        "model": {"depth": 12, "heads": 8, "opt": {"lr": 1e-3, "betas": [0.9, 0.99]}},  # dicts merge
        "grid": [44],                    # lists replace (no concatenation)
        "name": {"first": "x"},          # scalar replaced by a mapping
        "mixed": 3,                      # mapping replaced by a scalar
        "keep": 7,
        "extra": {"k": "v"},             # unknown keys are just data
    }
    assert load_config([b, a])["model"]["heads"] == 6      # left-to-right precedence
    assert load_config(str(a))["grid"] == [11, 22, 33]     # a single path is accepted
    assert load_config([]) == {}
    assert load_config(_write(tmp_path / "empty.yaml", "# only a comment\n")) == {}


def test_load_config_rejects_bad_files(tmp_path):
    with pytest.raises(ValueError, match="mapping"):
        load_config(_write(tmp_path / "list.yaml", "- 1\n- 2\n"))
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_yaml_aliases_do_not_share_containers(tmp_path):
    # YAML aliases return the *same* Python object; an override must not leak into siblings
    p = _write(tmp_path / "c.yaml", """
        base: &b {lr: 1, sched: [1, 2]}
        arm1: *b
        arm2: *b
    """)
    cfg = load_config(p, ["arm1.lr=5"])
    assert cfg["arm1"]["lr"] == 5 and cfg["arm2"]["lr"] == 1 and cfg["base"]["lr"] == 1
    cfg["arm1"]["sched"].append(3)
    assert cfg["arm2"]["sched"] == [1, 2]


def test_overrides_parse_values_as_yaml(tmp_path):
    p = _write(tmp_path / "c.yaml", """
        a: {b: {c: 1}}
        lr: 0.1
        name: x
        flag: false
        lst: [1]
        opt: {kind: adam}
    """)
    cfg = load_config(p, ["a.b.c=2.5", "a.b.c=3", "lr=1e-4", "name='007'", "flag=true",
                          "lst=[1, 2, {k: v}]", "opt=null", "+new.deep.key=hello world", "+top=1.0e5"])
    assert cfg["a"]["b"]["c"] == 3 and type(cfg["a"]["b"]["c"]) is int      # last override wins
    assert cfg["lr"] == 1e-4 and type(cfg["lr"]) is float                     # exponent float
    assert cfg["name"] == "007" and cfg["flag"] is True and cfg["opt"] is None
    assert cfg["lst"] == [1, 2, {"k": "v"}]
    assert cfg["new"] == {"deep": {"key": "hello world"}}                     # '+' creates the path
    assert cfg["top"] == 1e5 and type(cfg["top"]) is float
    assert load_config(p, ["+lr=2"])["lr"] == 2                               # '+' may also replace


def test_overrides_require_existing_path_unless_plus(tmp_path):
    p = _write(tmp_path / "c.yaml", "a: {b: {c: 1}}\nn: 3\n")
    for missing in ["a.b.d=1", "a.x.c=1", "zzz=1"]:
        with pytest.raises(ValueError, match="does not exist"):
            load_config(p, [missing])
    for through_scalar in ["a.b.c.d=1", "+a.b.c.d=1", "+n.sub=1"]:   # cannot descend into a scalar
        with pytest.raises(ValueError, match="not a mapping"):
            load_config(p, [through_scalar])
    for malformed in ["noequals", "=1", "+=1", "a..b=1", ".a=1", "a.=1"]:
        with pytest.raises(ValueError, match="must look like"):
            load_config(p, [malformed])
    with pytest.raises(ValueError, match="not valid YAML"):
        load_config(p, ["n=[1, 2"])
    with pytest.raises(ValueError, match="sequence"):
        load_config(p, "n=1")                     # a bare string is not a list of overrides
    with pytest.raises(ValueError, match="must be a string"):
        load_config(p, [3])


def test_exponent_floats_are_floats_and_pyyaml_is_untouched(tmp_path):
    p = _write(tmp_path / "c.yaml", "a: 1e-3\nb: 1.0e5\nc: -2E+2\nd: '1e-3'\ne: 1e3x\nf: 1.0e-5\n"
                                    "g: .5e-1\nh: 1_000e3\ni: ._e5\nj: 12\n")
    cfg = load_config(p)
    assert cfg == {"a": 1e-3, "b": 1e5, "c": -200.0, "d": "1e-3", "e": "1e3x", "f": 1e-5,
                   "g": 0.05, "h": 1e6, "i": "._e5", "j": 12}
    assert all(type(cfg[k]) is float for k in "abcfgh") and type(cfg["j"]) is int
    assert yaml.safe_load("a: 1e-3")["a"] == "1e-3"    # the global PyYAML loader is not modified


# ----------------------------------------------------------------------------- save / fingerprint / JSON
def test_save_config_round_trip(tmp_path):
    cfg = {"lr": 1e-4, "big": 1e20, "R": float("inf"), "s": "1e-3", "yes_str": "yes", "n": None,
           "flag": True, "lst": [1, 2.5, "x"], "nested": {"k": (1, 2), "p": Path("a") / "b"},
           "np": np.float64(0.25), "i": np.int64(3), "day": date(2026, 11, 17), "zh": "码率"}
    path = tmp_path / "sub" / "config.yaml"
    save_config(cfg, path)
    back = load_config(path)
    assert back["s"] == "1e-3" and back["yes_str"] == "yes"       # strings stay strings (quoted)
    assert back["lr"] == 1e-4 and back["big"] == 1e20 and back["R"] == float("inf")
    assert back["nested"] == {"k": [1, 2], "p": "a/b"} and back["np"] == 0.25 and back["i"] == 3
    assert back["day"] == date(2026, 11, 17) and back["zh"] == "码率"
    assert config_fingerprint(back) == config_fingerprint(cfg)
    assert not list(path.parent.glob("*.tmp"))                    # atomic write left no temp file
    with pytest.raises(ValueError, match="mapping"):
        save_config([1, 2], tmp_path / "x.yaml")


def test_config_fingerprint():
    a = {"x": 1, "y": {"b": [1, 2], "a": 0.5}}
    b = {"y": {"a": 0.5, "b": [1, 2]}, "x": 1}
    fp = config_fingerprint(a)
    assert re.fullmatch(r"[0-9a-f]{16}", fp)
    assert config_fingerprint(b) == fp                                           # key order
    assert config_fingerprint({"x": np.int64(1), "y": {"a": np.float64(0.5), "b": (1, 2)}}) == fp
    # pinned canonical form: sha256 of the compact, key-sorted JSON
    assert fp == hashlib.sha256(b'{"x":1,"y":{"a":0.5,"b":[1,2]}}').hexdigest()[:16]
    assert config_fingerprint({"x": 2, "y": {"a": 0.5, "b": [1, 2]}}) != fp
    assert config_fingerprint({"x": 1, "y": {"a": 0.5, "b": [2, 1]}}) != fp      # list order matters
    assert config_fingerprint({"x": 1.0, "y": {"a": 0.5, "b": [1, 2]}}) != fp    # 1 vs 1.0 differ
    with pytest.raises(ValueError, match="mapping"):
        config_fingerprint([1, 2])


@dataclasses.dataclass
class _Point:
    x: float
    tags: tuple
    arr: np.ndarray


def test_to_jsonable():
    obj = {
        "f32": np.float32(0.5), "i64": np.int64(3), "b": np.bool_(True), "s": np.str_("w"),
        "arr": np.array([[1.0, np.inf], [-np.inf, np.nan]]), "zero_d": np.array(2.0),
        "inf": float("inf"), "path": Path("a") / "b", "tup": (1, (2, 3)), "set": {10, 9, 1},
        "mixed_set": {1, "a"}, "dc": _Point(1.0, ("u",), np.arange(2)),
        "t": torch.tensor([1.5, 2.0]), "day": date(2026, 11, 17),
        (0, 1): "tuple key", 5: "int key", np.int64(6): "numpy key",
    }
    out = to_jsonable(obj)
    json.dumps(out, allow_nan=False)                                   # must never raise
    assert out["arr"] == [[1.0, "inf"], ["-inf", "nan"]] and out["inf"] == "inf"
    assert out["f32"] == 0.5 and type(out["i64"]) is int and out["b"] is True and type(out["s"]) is str
    assert out["zero_d"] == 2.0 and out["path"] == "a/b" and out["tup"] == [1, [2, 3]]
    assert out["set"] == [1, 9, 10] and sorted(out["mixed_set"], key=str) == [1, "a"]
    assert out["dc"] == {"x": 1.0, "tags": ["u"], "arr": [0, 1]}
    assert out["t"] == [1.5, 2.0] and out["day"] == "2026-11-17"
    assert out["(0, 1)"] == "tuple key" and out["5"] == "int key" and out["6"] == "numpy key"
    assert to_jsonable({"q": [{1, 2}, {2, 1}]}) == {"q": [[1, 2], [1, 2]]}   # deterministic sets
    with pytest.raises(TypeError, match="object"):
        to_jsonable({"x": object()})
    with pytest.raises(ValueError, match="collide"):
        to_jsonable({1: "a", "1": "b"})


# ----------------------------------------------------------------------------- Run
def test_run_lifecycle(tmp_path, capsys):
    reg, out = tmp_path / "registry.jsonl", tmp_path / "run"
    cfg = {"seed": 0, "model": {"lr": 1e-3, "dims": (32, 16)}}
    run = Run("unit_test_script", cfg, out, registry=reg)
    assert run.out_dir == out.resolve() and not run.resumed
    assert config_fingerprint(load_config(out / "config.yaml")) == config_fingerprint(cfg) == run.fingerprint

    run.logger.info("hello from the run")
    logging.getLogger("rwvae.some_library_module").warning("library warning")   # reaches log.txt too
    run.log_metrics(0, loss=np.float32(1.5), acc=0.25)
    run.log_metrics(None, loss=float("nan"))
    p = run.save_json("diag/result.json", {"A": np.eye(2), "inf": np.inf, "zh": "语义"})
    assert p == out.resolve() / "diag" / "result.json"
    assert json.loads(p.read_text(encoding="utf-8")) == {"A": [[1.0, 0.0], [0.0, 1.0]], "inf": "inf", "zh": "语义"}
    run.finish("ok", fid=np.float64(12.5), n=np.int64(3))

    m0, m1 = _lines(out / "metrics.jsonl")
    assert (m0["step"], m0["loss"], m0["acc"], m1["step"], m1["loss"]) == (0, 1.5, 0.25, None, "nan")
    assert datetime.fromisoformat(m0["time"]).utcoffset().total_seconds() == 0

    log = (out / "log.txt").read_text(encoding="utf-8")
    assert "hello from the run" in log and "library warning" in log and "status=ok" in log
    stdout = capsys.readouterr().out
    assert stdout.count("hello from the run") == 1                       # stdout, exactly once

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    (rec,) = _lines(reg)
    assert rec == summary and set(rec) == RECORD_KEYS
    assert rec["status"] == "ok" and rec["script"] == "unit_test_script"
    assert rec["summary"] == {"fid": 12.5, "n": 3} and rec["config_fingerprint"] == run.fingerprint
    assert Path(rec["out_dir"]) == out.resolve() and rec["env"] == env_info()
    assert rec["git_commit"] is None or isinstance(rec["git_commit"], str)
    assert datetime.fromisoformat(rec["time"]).utcoffset().total_seconds() == 0

    os.remove(out / "log.txt")      # finish() released the log file (an open file cannot be deleted on Windows)
    with pytest.raises(RuntimeError, match="already"):
        run.finish("ok")
    assert len(_lines(reg)) == 1


def test_run_existing_directory_semantics(tmp_path):
    reg, out = tmp_path / "registry.jsonl", tmp_path / "run"
    first = Run("s", {"a": 1}, out, registry=reg)
    first.log_metrics(0, x=1.0)
    first.finish("ok")
    (out / "artifact.bin").write_bytes(b"script-owned")

    with pytest.raises(FileExistsError, match="resume"):
        Run("s", {"a": 1}, out, registry=reg)

    resumed = Run("s", {"a": 1}, out, registry=reg, resume=True)      # same fingerprint
    assert resumed.resumed and not (out / "summary.json").exists()    # stale summary removed
    resumed.log_metrics(1, x=2.0)
    resumed.finish("ok")
    assert [m["step"] for m in _lines(out / "metrics.jsonl")] == [0, 1]   # appended, not restarted

    with pytest.raises(ValueError, match="fingerprint"):
        Run("s", {"a": 2}, out, registry=reg, resume=True)
    assert load_config(out / "config.yaml") == {"a": 1}                # refused runs touch nothing
    changed = Run("s", {"a": 2}, out, registry=reg, resume=True, overwrite=True)
    assert changed.resumed and load_config(out / "config.yaml") == {"a": 2}
    changed.finish("ok")
    assert len(_lines(out / "metrics.jsonl")) == 2

    fresh = Run("s", {"a": 3}, out, registry=reg, overwrite=True)       # fresh run in place
    assert not fresh.resumed and not (out / "metrics.jsonl").exists()
    assert (out / "artifact.bin").read_bytes() == b"script-owned"      # other files are never deleted
    fresh.finish("ok")

    nothing = Run("s", {"a": 1}, tmp_path / "new", registry=reg, resume=True)   # nothing to resume
    assert not nothing.resumed
    nothing.finish("ok")

    fps = [config_fingerprint({"a": a}) for a in (1, 1, 2, 3, 1)]
    assert [r["config_fingerprint"] for r in _lines(reg)] == fps
    assert (out / "log.txt").read_text(encoding="utf-8").count("status=ok") == 4   # log is appended


def test_run_input_validation(tmp_path):
    reg = tmp_path / "registry.jsonl"
    with pytest.raises(ValueError, match="script"):
        Run("", {}, tmp_path / "r0", registry=reg)
    with pytest.raises(ValueError, match="mapping"):
        Run("s", [1, 2], tmp_path / "r1", registry=reg)
    with pytest.raises(ValueError, match="out_dir"):
        Run("s", {}, None, registry=reg)                 # e.g. an unset --out passed straight through
    assert not (tmp_path / "r0").exists() and not (tmp_path / "r1").exists()

    run = Run("s", {"a": 1}, tmp_path / "r", registry=reg)
    for bad_step in (1.5, True, "3"):
        with pytest.raises(ValueError, match="step"):
            run.log_metrics(bad_step, x=1.0)
    with pytest.raises(ValueError, match="reserved"):
        run.log_metrics(0, time=1.0)
    for bad_name in ("../escape.json", str(tmp_path / "abs.json"), "", ".", "a/../../b.json"):
        with pytest.raises(ValueError, match="relative path"):
            run.save_json(bad_name, {})
    with pytest.raises(TypeError):
        run.save_json("bad.json", {"x": object()})
    with pytest.raises(ValueError, match="status"):
        run.finish("")
    run.finish("failed", error="boom")
    (rec,) = _lines(reg)
    assert rec["status"] == "failed" and rec["summary"] == {"error": "boom"}


def test_get_logger_is_idempotent(tmp_path, capsys):
    name = "rwvae_test_runtime_logger"          # not under "rwvae": independent of Run handlers
    f1, f2 = tmp_path / "a" / "log1.txt", tmp_path / "log2.txt"
    logger = get_logger(name, f1)
    try:
        assert get_logger(name, f1) is logger and len(logger.handlers) == 2   # stdout + file
        get_logger(name, str(f1))
        get_logger(name)                                                      # keeps the file handler
        assert len(logger.handlers) == 2
        logger.info("first")
        get_logger(name, f2)                                                  # redirect: still 2 handlers
        assert len(logger.handlers) == 2 and not logger.propagate
        logger.info("second")
    finally:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()
    assert f1.read_text(encoding="utf-8").count("first") == 1 and "second" not in f1.read_text(encoding="utf-8")
    assert f2.read_text(encoding="utf-8").count("second") == 1
    stdout = capsys.readouterr().out
    assert stdout.count("first") == 1 and stdout.count("second") == 1


# ----------------------------------------------------------------------------- seeds / devices
@pytest.fixture
def restore_torch_determinism():
    env = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    state = (torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.benchmark)
    yield
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if env is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = env
    torch.use_deterministic_algorithms(state[0], warn_only=state[1])
    torch.backends.cudnn.benchmark = state[2]


def _draws():
    out = [random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist()]
    if torch.cuda.is_available():
        out.append(torch.rand(3, device="cuda").tolist())
    return out


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = _draws()
    seed_everything(np.int64(123))
    b = _draws()
    seed_everything(124)
    c = _draws()
    assert a == b
    assert all(x != y for x, y in zip(a, c))
    for bad in (-1, 2 ** 32, 1.5, True, "1", None):
        with pytest.raises(ValueError, match="seed"):
            seed_everything(bad)


def test_seed_everything_deterministic_flags(restore_torch_determinism):
    seed_everything(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_select_device(monkeypatch):
    assert select_device("cpu") == torch.device("cpu")
    assert select_device().type == ("cuda" if torch.cuda.is_available() else "cpu")
    for bad in ("gpu", "cuda:", "cuda:x", "cuda:-1", "", "CPU", "cuda:0:1", " cuda", None):
        with pytest.raises(ValueError, match="device must be"):
            select_device(bad)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto") == torch.device("cpu")
    for pref in ("cuda", "cuda:0"):
        with pytest.raises(ValueError, match="CUDA is not available"):
            select_device(pref)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert select_device("cuda") == torch.device("cuda")
    assert select_device("cuda:1") == torch.device("cuda", 1)
    with pytest.raises(ValueError, match="only 2 CUDA"):
        select_device("cuda:2")


# ----------------------------------------------------------------------------- provenance
def test_sha256_file_matches_hashlib(tmp_path):
    data = np.random.default_rng(0).bytes(3 * 1024 + 17)
    p = tmp_path / "blob.bin"
    p.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_file(p) == expected
    assert sha256_file(str(p), chunk=1000) == expected          # several chunks + a ragged tail
    assert sha256_file(p, chunk=1) == expected
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert sha256_file(empty) == hashlib.sha256(b"").hexdigest()
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="chunk"):
            sha256_file(p, chunk=bad)


def test_git_commit_returns_hash_or_none(tmp_path):
    commit = git_commit()
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}([0-9a-f]{24})?", commit)
    assert git_commit(tmp_path / "does-not-exist") is None


def test_env_info():
    info = env_info()
    assert {"python", "platform", "torch", "numpy", "cuda_available", "cuda_version", "gpus"} <= set(info)
    assert info["cuda_available"] == torch.cuda.is_available()
    assert len(info["gpus"]) == (torch.cuda.device_count() if info["cuda_available"] else 0)
    json.dumps(info, allow_nan=False)


# ----------------------------------------------------------------------------- CLI
def test_common_args_and_parse_config(tmp_path):
    a = _write(tmp_path / "a.yaml", "seed: 1\nx: {y: 2}\n")
    b = _write(tmp_path / "b.yaml", "x: {z: 3}\n")
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(["--config", str(a), "--config", str(b), "--set", "x.y=5",
                              "--set", "+x.w=[1, 2]", "--seed", "7", "--out", "o", "--overwrite"])
    assert parse_config_from_args(args) == {"seed": 7, "x": {"y": 5, "z": 3, "w": [1, 2]}}
    assert (args.out, args.device, args.overwrite, args.resume) == ("o", "auto", True, False)

    defaults = parser.parse_args([])
    assert (defaults.config, defaults.set, defaults.seed, defaults.out) == (None, None, None, None)
    assert parse_config_from_args(defaults) == {}
    no_seed = parser.parse_args(["--config", str(a), "--device", "cpu", "--resume"])
    assert parse_config_from_args(no_seed) == {"seed": 1, "x": {"y": 2}}
    assert no_seed.device == "cpu" and no_seed.resume
