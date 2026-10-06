"""Shared paths, hashing and atomic-write helpers for the v30 study.

v30 studies exclusive six-group combinations of four metabolic laboratory
findings in 19-39 year old KNHANES respondents. This module has no
study logic; it only fixes where things live and how provenance is recorded.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/knhanes"
GUIDE_DIR = ROOT / "data/reference/knhanes"
OUT_ROOT = ROOT / "artifacts/clinical_v30"
PROTOCOL = ROOT / "docs/V30_PROTOCOL.md"
PRIVATE_DIR_NAME = "private"  # row-level files; never published

ALL_YEARS = tuple(range(2015, 2025))
PERIODS = {"P1_2015_2019": tuple(range(2015, 2020)), "P2_2020_2024": tuple(range(2020, 2025))}
# KNHANES phases (기수) by survey year; 2015 belongs to the 6th phase.
PHASE = {2015: 6, 2016: 7, 2017: 7, 2018: 7, 2019: 8, 2020: 8, 2021: 8,
         2022: 9, 2023: 9, 2024: 9}


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, default=_default).encode()).hexdigest()


def sha256_array(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(np.asarray(array))
        digest.update(str(value.dtype).encode() + str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return rel(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def _clean(value):
    """Replace non-finite floats with None so JSON stays strict."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    return value


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace(os.sep, "/")
    except ValueError:
        return str(path).replace(os.sep, "/")


def write_json(path: Path, value, *, overwrite: bool = False) -> str:
    """Atomic strict JSON write; refuses to replace different content unless asked."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(value), ensure_ascii=False, indent=1, allow_nan=False,
                      default=_default) + "\n"
    if path.exists() and not overwrite:
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Refusing to overwrite different content: {rel(path)}")
        return sha256_file(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # per-process: workers may race
    temporary.write_text(text, encoding="utf-8")
    replace_retry(temporary, path)
    return sha256_file(path)


def replace_retry(source: Path, target: Path, attempts: int = 20) -> None:
    """os.replace with retries: on Windows a concurrent reader can briefly lock the target."""
    import time as _time
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            _time.sleep(0.2 * (attempt + 1))


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: Path, text: str, *, overwrite: bool = True) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Refusing to overwrite different content: {rel(path)}")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    replace_retry(temporary, path)
    return sha256_file(path)


def runtime_versions() -> dict:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "torch", "lightgbm",
             "pyreadstat", "matplotlib", "statsmodels")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": versions}


def source_hashes(extra: tuple[str, ...] = ()) -> dict[str, str]:
    """Hashes of every v30 module plus the reused v21/v23/v24 sources."""
    files = sorted((ROOT / "metabolic").glob("v30_*.py"))
    reused = ("metabolic/event_field_v21.py", "metabolic/event_field_v23.py",
              "metabolic/clinical_v23_distribution.py",
              "metabolic/clinical_v24_distribution.py", "metabolic/data.py")
    paths = [*files, *(ROOT / name for name in reused), *(ROOT / name for name in extra)]
    return {rel(path): sha256_file(path) for path in paths if Path(path).exists()}


def protocol_hash() -> str | None:
    return sha256_file(PROTOCOL) if PROTOCOL.exists() else None


def suppress_small(count: int, minimum: int = 5):
    """Public small-cell suppression: counts 1..minimum-1 are shown as '<5'."""
    count = int(count)
    return f"<{minimum}" if 0 < count < minimum else count
