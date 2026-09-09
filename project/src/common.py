#!/usr/bin/env python3
"""Shared configuration, provenance, logging, and IO utilities.

Implements prompt2.txt sections 39 (reproducibility), 44 (software
engineering), and the "PROJECT_AUDIT" / "UNKNOWN / NOT AVAILABLE"
principles (section 45).
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

SOFTWARE = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
}

try:
    import rdkit
    SOFTWARE["rdkit"] = rdkit.__version__
except Exception:  # pragma: no cover
    SOFTWARE["rdkit"] = "NOT_AVAILABLE"

try:
    import sklearn
    SOFTWARE["scikit-learn"] = sklearn.__version__
except Exception:
    SOFTWARE["scikit-learn"] = "NOT_AVAILABLE"

try:
    import scipy
    SOFTWARE["scipy"] = scipy.__version__
except Exception:
    SOFTWARE["scipy"] = "NOT_AVAILABLE"

try:
    import matplotlib
    SOFTWARE["matplotlib"] = matplotlib.__version__
except Exception:
    SOFTWARE["matplotlib"] = "NOT_AVAILABLE"

try:
    import shap
    SOFTWARE["shap"] = shap.__version__
except Exception:
    SOFTWARE["shap"] = "NOT_AVAILABLE"

try:
    import networkx
    SOFTWARE["networkx"] = networkx.__version__
except Exception:
    SOFTWARE["networkx"] = "NOT_AVAILABLE"

# REINVENT / AF3 / Boltz are external; presence determined at runtime.
SOFTWARE["reinvent"] = "NOT_AVAILABLE"
SOFTWARE["alphafold3"] = "NOT_AVAILABLE"
SOFTWARE["boltz"] = "NOT_AVAILABLE"


def sha256_file(path: UnionPaths) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_dataframe(df: pd.DataFrame) -> str:
    return sha256_text(df.to_csv(index=False))


def sha256_config(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, default=str))


UnionPaths = str


class ProjectConfig:
    """Loads config.yaml and exposes typed accessors with validation."""

    def __init__(self, config_path: str) -> None:
        self.path = str(config_path)
        with open(config_path, encoding="utf-8") as f:
            self.raw = yaml.safe_load(f)
        self.validate()
        self.project = self.raw["project"]
        self.data = self.raw["data"]
        self.seeds = self.raw["seeds"]
        self.root = Path(self.project["root"])

    def validate(self) -> None:
        required_top = ["project", "data", "seeds", "models", "validation"]
        for key in required_top:
            if key not in self.raw:
                raise ValueError(f"config missing required section: {key}")
        if "raw_csv" not in self.raw["data"]:
            raise ValueError("config data.raw_csv required")

    def resolve(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def seed(self, key: str = "global") -> int:
        return int(self.seeds.get(key, self.seeds.get("global", 42)))

    @property
    def cycle(self) -> int:
        return int(self.project.get("cycle", 0))

    def as_jsonable(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)


class ManagedLogger:
    """File + console logger that records every major step."""

    def __init__(self, name: str, log_dir: str, level: int = logging.INFO) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            fh = logging.FileHandler(
                os.path.join(log_dir, f"{name}.log"), encoding="utf-8"
            )
            fh.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            self.logger.addHandler(fh)
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(ch)

    def step(self, msg: str) -> None:
        self.logger.info("=== %s ===", msg)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warn(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_provenance(module: str, config: ProjectConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Build a provenance record (prompt2 sect. 24, 39)."""
    return {
        "generated_at": utcnow(),
        "module": module,
        "python": SOFTWARE.get("python"),
        "rdkit": SOFTWARE.get("rdkit"),
        "scikit_learn": SOFTWARE.get("scikit-learn"),
        "pandas": SOFTWARE.get("pandas"),
        "numpy": SOFTWARE.get("numpy"),
        "seeds": config.seeds,
        "config_hash": sha256_config(config.raw),
        "input_hashes": {k: v for k, v in inputs.items()},
        "git_commit": git_commit(),
        "config_path": config.path,
    }


def git_commit() -> str:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.getcwd())),
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "NOT_A_GIT_REPO_OR_NOT_COMMITTED"


def save_manifest(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def save_df(df: pd.DataFrame, path: str, index: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)


def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def available_in(paths: Dict[str, Optional[str]]) -> Dict[str, bool]:
    return {k: (v is not None and os.path.exists(v)) for k, v in paths.items()}


def check_external_tools() -> Dict[str, Any]:
    """Detect external scientific tools without importing them."""
    import shutil
    results = {
        "reinvent": shutil.which("reinvent") is not None or shutil.which("reinvent4") is not None,
        "openeye": shutil.which("oeLicense") is not None,
    }
    return results