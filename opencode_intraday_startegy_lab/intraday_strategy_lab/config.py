from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install dependencies with `pip install -r requirements.txt`.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_config_dir(config_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    root = config_dir or PROJECT_ROOT / "config"
    return {
        "dhan": load_yaml(root / "dhan.yaml"),
        "universe": load_yaml(root / "universe.yaml"),
        "backtest": load_yaml(root / "backtest.yaml"),
        "optimization": load_yaml(root / "optimization.yaml"),
    }


def get_env_setting(config: dict[str, Any], key: str, default: str = "") -> str:
    env_name = str(config.get(f"{key}_env", ""))
    if env_name:
        value = os.getenv(env_name, "")
        if value:
            return value
    return str(config.get(key, default) or default)


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
