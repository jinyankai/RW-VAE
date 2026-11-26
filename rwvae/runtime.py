"""Run-time utilities shared by the RW-VAE v2 entry points (ARCHITECTURE_V2.md Sec. 1, 3.1).

- config: YAML files deep-merged left to right plus dotted ``a.b.c=VALUE`` overrides
  (`load_config`, `save_config`, `config_fingerprint`);
- provenance for artifact ``meta`` (plan v2.1 Sec. 3.2): `sha256_file`, `git_commit`, `git_dirty`, `env_info`;
- reproducibility: `seed_everything`, `select_device`;
- run directory: `Run` owns ``config.yaml``, ``log.txt``, ``metrics.jsonl`` and ``summary.json``
  in its out_dir and appends one record per finished run to ``manifests/run_registry.jsonl``;
- CLI: `add_common_args`, `parse_config_from_args`.

Every JSON output goes through `to_jsonable`, so numpy values, paths and non-finite floats
never break ``json.dumps(..., allow_nan=False)``.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path, PurePath
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "manifests" / "run_registry.jsonl"
PACKAGE_LOGGER = "rwvae"  # Run attaches its handlers here, so library loggers (rwvae.*) reach log.txt

# ----------------------------------------------------------------------------- YAML
# PyYAML implements YAML 1.1, where "1e-3" and "1.0e5" load as *strings* (a float needs a dot
# and a signed exponent). Configs are written YAML 1.2 / JSON style, so exponent forms are
# resolved as floats -- in the loader and in the dumper, so that the string "1e-3" is quoted on
# save and every config round-trips exactly. Everything else is plain yaml.safe_load semantics.
_EXP_FLOAT = re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9][0-9_]*)[eE][-+]?[0-9]+$")


class _Loader(yaml.SafeLoader):
    pass


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:  # always write plain trees, never &anchors
        return True


for _cls in (_Loader, _Dumper):
    _cls.add_implicit_resolver("tag:yaml.org,2002:float", _EXP_FLOAT, list("-+0123456789."))


# ----------------------------------------------------------------------------- JSON
def _convert(obj: Any, json_safe: bool) -> Any:
    """Plain-Python copy of obj; json_safe=True also maps non-finite floats/dates to strings."""
    if isinstance(obj, np.generic):
        obj = obj.item()
    elif isinstance(obj, np.ndarray):
        obj = obj.tolist()
    elif isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().tolist()
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if json_safe and not math.isfinite(obj):
            return "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
        return obj
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if isinstance(obj, date):  # datetime is a subclass; YAML stores both natively
        return obj.isoformat() if json_safe else obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _convert(getattr(obj, f.name), json_safe) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = k if type(k) is str else str(k)
            if key in out:
                raise ValueError(f"mapping keys collide after conversion to str: {key!r}")
            out[key] = _convert(v, json_safe)
        return out
    if isinstance(obj, (list, tuple)):
        return [_convert(v, json_safe) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [_convert(v, json_safe) for v in obj]
        try:
            return sorted(items)
        except TypeError:  # mixed element types: any fixed total order will do
            return sorted(items, key=lambda v: (type(v).__name__, repr(v)))
    raise TypeError(f"cannot convert {type(obj).__name__} to JSON: {obj!r:.80}")


def to_jsonable(obj: Any) -> Any:
    """Recursively convert obj into plain JSON types.

    numpy scalars/arrays and torch tensors -> Python scalars/lists; Path -> POSIX string;
    tuple -> list; set -> sorted list; dataclass -> dict of its fields; date -> ISO string;
    mapping keys -> str (a collision such as {1: .., "1": ..} raises ValueError); non-finite
    floats -> "inf" / "-inf" / "nan". Afterwards json.dumps(..., allow_nan=False) cannot fail.
    Any other type raises TypeError, as json.dumps would.
    """
    return _convert(obj, json_safe=True)


def _canonical_json(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text with LF newlines to a sibling temp file, then os.replace it over path."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def _append_jsonl(path: Path, record: Mapping) -> None:
    """Append one compact JSON line with a single write call."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(to_jsonable(record), ensure_ascii=False, allow_nan=False)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------- config
def _check_mapping(cfg: Any) -> None:
    if not isinstance(cfg, Mapping):
        raise ValueError(f"a config must be a mapping, got {type(cfg).__name__}: {cfg!r:.80}")


def _fresh(obj: Any) -> Any:
    """Copy the dict/list skeleton, so no container is shared (YAML aliases share objects)."""
    if isinstance(obj, dict):
        return {k: _fresh(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fresh(v) for v in obj]
    return obj


def _deep_merge(base: dict, new: dict) -> dict:
    """base updated by new: two mappings merge recursively, any other value replaces."""
    out = dict(base)
    for k, v in new.items():
        both_dicts = isinstance(out.get(k), dict) and isinstance(v, dict)
        out[k] = _deep_merge(out[k], v) if both_dicts else _fresh(v)
    return out


def _apply_override(cfg: dict, item: str) -> None:
    """Set cfg[a][b][c] = yaml(VALUE) for "a.b.c=VALUE"; "+a.b.c=VALUE" may create keys."""
    if not isinstance(item, str):
        raise ValueError(f"an override must be a string, got {item!r}")
    create = item.startswith("+")
    key, sep, raw = (item[1:] if create else item).partition("=")
    parts = key.split(".")
    if not sep or not all(parts):
        raise ValueError(f"override must look like 'a.b.c=VALUE' or '+a.b.c=VALUE', got {item!r}")
    try:
        value = yaml.load(raw, Loader=_Loader)
    except yaml.YAMLError as e:
        raise ValueError(f"override {item!r}: VALUE is not valid YAML ({e})") from e
    node = cfg
    for i, part in enumerate(parts):
        path = ".".join(parts[: i + 1])
        if part not in node and not create:
            raise ValueError(f"override {item!r}: key {path!r} does not exist "
                             f"(write '+{item}' to create it)")
        if i == len(parts) - 1:
            node[part] = value
            return
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"override {item!r}: {path!r} is a {type(node).__name__}, not a mapping")


def load_config(paths: str | os.PathLike | Sequence[str | os.PathLike],
                overrides: Sequence[str] = ()) -> dict:
    """Deep-merge YAML files left to right, then apply dotted overrides; returns plain dicts.

    Mappings merge recursively; scalars and lists replace. An empty file is {}. Unknown keys
    are fine (a config is data). Each override "a.b.c=VALUE" (VALUE parsed as YAML) must name
    an existing key, otherwise ValueError; "+a.b.c=VALUE" creates missing keys.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    if isinstance(overrides, str):
        raise ValueError(f"overrides must be a sequence of 'key=value' strings, got the string {overrides!r}")
    cfg: dict = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = yaml.load(f, Loader=_Loader)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"config file {p} must hold a mapping at top level, got {type(data).__name__}")
        cfg = _deep_merge(cfg, data)
    for item in overrides:
        _apply_override(cfg, item)
    return cfg


def save_config(cfg: Mapping, path: str | os.PathLike) -> None:
    """Write cfg as YAML (atomically) that `load_config` reads back with the same fingerprint."""
    _check_mapping(cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(_convert(cfg, json_safe=False), Dumper=_Dumper, sort_keys=False, allow_unicode=True)
    _write_text_atomic(path, text)


def config_fingerprint(cfg: Mapping) -> str:
    """First 16 hex digits of sha256 over the canonical JSON of `to_jsonable(cfg)` (sorted keys,
    compact separators, ASCII). Key order and numpy-vs-Python scalars do not matter; 1 != 1.0."""
    _check_mapping(cfg)
    return hashlib.sha256(_canonical_json(cfg).encode("ascii")).hexdigest()[:16]


# ----------------------------------------------------------------------------- provenance
def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    """Hex sha256 of a file, streamed in blocks of `chunk` bytes."""
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk <= 0:
        raise ValueError(f"chunk must be a positive int, got {chunk!r}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit(repo_dir: str | os.PathLike | None = None) -> str | None:
    """HEAD commit of the git repo at repo_dir (default: this repo), or None when git or the
    repo is unavailable; never raises. Uncommitted changes are reported by `git_dirty`."""
    repo = REPO_ROOT if repo_dir is None else repo_dir
    try:
        res = subprocess.run(["git", "-C", os.fspath(repo), "rev-parse", "HEAD"],
                             capture_output=True, timeout=10, check=True)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return res.stdout.decode("ascii", errors="replace").strip() or None


def git_dirty(repo_dir: str | os.PathLike | None = None) -> bool | None:
    """True if the work tree at repo_dir (default: this repo) differs from HEAD: modified or staged
    tracked files, or untracked files that .gitignore does not exclude. None when git or the repo is
    unavailable; never raises. A run with git_dirty=True is not reproducible from git_commit alone."""
    repo = REPO_ROOT if repo_dir is None else repo_dir
    try:
        res = subprocess.run(["git", "-C", os.fspath(repo), "status", "--porcelain", "--untracked-files=normal"],
                             capture_output=True, timeout=30, check=True)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return bool(res.stdout.strip())


def env_info() -> dict:
    """Software/hardware environment for run records: python, platform, torch, numpy, CUDA, GPUs."""
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,  # CUDA version torch was built with (None: CPU build)
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
    }


# ----------------------------------------------------------------------------- reproducibility
def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python `random`, NumPy's global RNG and torch (CPU and every CUDA device).

    deterministic=True also calls torch.use_deterministic_algorithms(True, warn_only=True),
    sets cudnn.benchmark=False and sets CUBLAS_WORKSPACE_CONFIG=:4096:8 unless already set
    (needed for deterministic cuBLAS; effective only before the first cuBLAS call). Library
    code still takes explicit seeds/generators; this only pins the global generators.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2 ** 32:
        raise ValueError(f"seed must be an integer in [0, 2**32), got {seed!r}")
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds the CPU generator and all CUDA generators
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


_CUDA_DEVICE = re.compile(r"cuda(?::([0-9]+))?")


def select_device(pref: str = "auto") -> torch.device:
    """"auto" -> cuda if available else cpu; "cpu"; "cuda" / "cuda:N". Asking for CUDA when it
    is unavailable (or for a missing index) raises ValueError -- never a silent CPU fallback."""
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cpu":
        return torch.device("cpu")
    match = _CUDA_DEVICE.fullmatch(pref) if isinstance(pref, str) else None
    if match is None:
        raise ValueError(f"device must be 'auto', 'cpu', 'cuda' or 'cuda:N', got {pref!r}")
    if not torch.cuda.is_available():
        raise ValueError(f"device {pref!r} requested but CUDA is not available")
    if match.group(1) is None:
        return torch.device("cuda")
    index, count = int(match.group(1)), torch.cuda.device_count()
    if index >= count:
        raise ValueError(f"device {pref!r} requested but only {count} CUDA device(s) are visible")
    return torch.device("cuda", index)


# ----------------------------------------------------------------------------- logging
# local time with UTC offset, so log lines can be matched to the UTC times in metrics/registry
_FORMATTER = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S%z")


class _StdoutHandler(logging.StreamHandler):
    """Writes to the *current* sys.stdout (like logging's own _StderrHandler), so it keeps
    working when stdout is replaced after the handler was created."""

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self):
        return sys.stdout


def _remove_handlers(logger: logging.Logger, handlers: Sequence[logging.Handler]) -> None:
    for h in handlers:
        logger.removeHandler(h)
        h.close()


def _same_file(a: str | os.PathLike, b: str | os.PathLike) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def get_logger(name: str, log_file: str | os.PathLike | None = None) -> logging.Logger:
    """INFO logger writing to stdout and, if given, to log_file (UTF-8, appended).

    Idempotent: repeated calls never add duplicate handlers. A logger writes to at most one
    file: another log_file replaces (and closes) the previous file handler. propagate is off,
    so a configured root logger cannot print records twice. Library modules should use
    logging.getLogger(__name__) and leave the handlers to the entry point (`Run`).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(h, _StdoutHandler) for h in logger.handlers):
        handler = _StdoutHandler()
        handler.setFormatter(_FORMATTER)
        logger.addHandler(handler)
    if log_file is not None:
        files = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        if not any(_same_file(h.baseFilename, log_file) for h in files):
            _remove_handlers(logger, files)
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setFormatter(_FORMATTER)
            logger.addHandler(handler)
    return logger


# ----------------------------------------------------------------------------- run directory
class Run:
    """One script run. Owns out_dir/{config.yaml, log.txt, metrics.jsonl, summary.json}.

    If out_dir already holds a run (has config.yaml):
      * neither flag    -> FileExistsError (nothing is touched);
      * resume=True     -> continue it: metrics.jsonl and log.txt are appended to. The saved
                           config must have the same fingerprint, else ValueError, unless
                           overwrite=True (then the new config replaces the saved one);
      * overwrite=True  -> fresh run in place: config.yaml rewritten, metrics.jsonl restarted.
    A stale summary.json is removed at start. Nothing else in out_dir is ever deleted, and a
    directory without config.yaml is used as is (resume=True then starts fresh; see `resumed`).

    Logging goes to stdout and out_dir/log.txt through the "rwvae" package logger, so library
    loggers (logging.getLogger(__name__) in rwvae.*) are captured too. finish() writes
    summary.json, appends the same record to the registry and closes log.txt; to record a
    failure call finish("failed", error=...). Create one Run per process (rank 0 under DDP).
    """

    def __init__(self, script: str, cfg: Mapping, out_dir: str | os.PathLike, *,
                 registry: str | os.PathLike | None = None, overwrite: bool = False,
                 resume: bool = False):
        if not isinstance(script, str) or not script:
            raise ValueError(f"script must be a non-empty string, got {script!r}")
        if out_dir is None:
            raise ValueError("out_dir is None: pass --out or give the run an output directory")
        self.script = script
        self.fingerprint = config_fingerprint(cfg)  # validates cfg before anything is written
        self.out_dir = Path(out_dir).resolve()
        self.registry = DEFAULT_REGISTRY if registry is None else Path(registry).resolve()
        cfg_path = self.out_dir / "config.yaml"
        existed = cfg_path.exists()
        saved_fp = None
        if existed and resume:
            saved_fp = config_fingerprint(load_config(cfg_path))
            if saved_fp != self.fingerprint and not overwrite:
                raise ValueError(
                    f"cannot resume {self.out_dir}: its config.yaml has fingerprint {saved_fp} but the "
                    f"current config has {self.fingerprint}; resume with the original config, pass "
                    f"overwrite=True (--overwrite) to continue under the new one, or choose another out_dir")
        elif existed and not overwrite:
            raise FileExistsError(
                f"{self.out_dir} already holds a run (config.yaml); pass resume=True (--resume) to "
                f"continue it, overwrite=True (--overwrite) to replace it, or choose another out_dir")
        self.resumed = bool(existed and resume)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        if existed:
            (self.out_dir / "summary.json").unlink(missing_ok=True)
            if not self.resumed:
                (self.out_dir / "metrics.jsonl").unlink(missing_ok=True)
        save_config(cfg, cfg_path)

        self._log_path = self.out_dir / "log.txt"
        get_logger(PACKAGE_LOGGER, self._log_path)
        self.logger = logging.getLogger(f"{PACKAGE_LOGGER}.{Path(script).stem}")
        self._commit, self._dirty, self._env = git_commit(), git_dirty(), env_info()
        self._finished = False

        mode = "resume" if self.resumed else ("overwrite" if existed else "new")
        self.logger.info("run start (%s): script=%s out_dir=%s config=%s git=%s dirty=%s", mode, script,
                         self.out_dir.as_posix(), self.fingerprint, self._commit, self._dirty)
        self.logger.info("argv: %s", " ".join(sys.argv))
        self.logger.info("env: %s", json.dumps(self._env, ensure_ascii=False))
        if resume and not existed:
            self.logger.warning("resume requested but %s holds no run; starting a fresh one", self.out_dir)
        if saved_fp is not None and saved_fp != self.fingerprint:
            self.logger.warning("resuming under a new config (saved fingerprint %s) because overwrite=True",
                                saved_fp)

    def _path_in_run(self, name: str) -> Path:
        rel = Path(name) if isinstance(name, str) else None
        if rel is None or not rel.parts or rel.anchor or ".." in rel.parts:
            raise ValueError(f"name must be a relative path inside the run directory, got {name!r}")
        return self.out_dir / rel

    def log_metrics(self, step: int | None, **scalars: Any) -> None:
        """Append {"time": ISO-8601 UTC, "step": step, **scalars} as one line to metrics.jsonl."""
        if step is not None and (isinstance(step, bool) or not isinstance(step, (int, np.integer))):
            raise ValueError(f"step must be an int or None, got {step!r}")
        if "time" in scalars:
            raise ValueError("'time' is a reserved metrics field")
        record = {"time": _utc_now(), "step": None if step is None else int(step), **scalars}
        _append_jsonl(self.out_dir / "metrics.jsonl", record)

    def save_json(self, name: str, obj: Any) -> Path:
        """Write to_jsonable(obj) as indented JSON to out_dir/name (atomic); returns the path."""
        path = self._path_in_run(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(path, json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False,
                                            allow_nan=False) + "\n")
        return path

    def finish(self, status: str = "ok", **summary: Any) -> None:
        """Write summary.json and append the same record to the run registry, then close log.txt.

        Record: time (ISO-8601 UTC), script, status, out_dir, config_fingerprint, git_commit,
        git_dirty (`git_dirty()` at start), env (`env_info()` at start), summary.
        """
        if self._finished:
            raise RuntimeError(f"finish() was already called for the run in {self.out_dir}")
        if not isinstance(status, str) or not status:
            raise ValueError(f"status must be a non-empty string, got {status!r}")
        record = to_jsonable({
            "time": _utc_now(), "script": self.script, "status": status,
            "out_dir": self.out_dir, "config_fingerprint": self.fingerprint,
            "git_commit": self._commit, "git_dirty": self._dirty, "env": self._env, "summary": summary,
        })
        _write_text_atomic(self.out_dir / "summary.json",
                           json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        _append_jsonl(self.registry, record)
        self._finished = True
        self.logger.info("run finished: status=%s", status)
        package = logging.getLogger(PACKAGE_LOGGER)
        mine = [h for h in package.handlers
                if isinstance(h, logging.FileHandler) and _same_file(h.baseFilename, self._log_path)]
        _remove_handlers(package, mine)  # release log.txt (Windows cannot delete open files)


# ----------------------------------------------------------------------------- CLI
def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags shared by all scripts/0x_*.py entry points (ARCHITECTURE_V2.md Sec. 4)."""
    parser.add_argument("--config", action="append", default=None, metavar="YAML",
                        help="YAML config file; repeatable, later files override earlier ones")
    parser.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                        help="override an existing config key with a YAML value, e.g. "
                             "--set stats.ridge_rel=1e-3; '+KEY=VALUE' creates a new key; repeatable")
    parser.add_argument("--out", default=None, metavar="DIR", help="run output directory")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    parser.add_argument("--seed", type=int, default=None, help="override cfg['seed']")
    parser.add_argument("--overwrite", action="store_true",
                        help="reuse an --out that already holds a run (its config is replaced)")
    parser.add_argument("--resume", action="store_true",
                        help="continue the run in --out (its saved config must match)")


def parse_config_from_args(args: argparse.Namespace) -> dict:
    """load_config(args.config or [], args.set or []); a --seed given on the CLI sets cfg["seed"]."""
    cfg = load_config(args.config or [], args.set or [])
    if args.seed is not None:
        cfg["seed"] = args.seed
    return cfg
