"""Load configuration files from configs/."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str) -> dict:
    with open(ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def params() -> dict:
    return load_yaml("params.yaml")


def windows() -> dict:
    return load_yaml("treatment_windows.yaml")


def datasets() -> dict:
    return load_yaml("datasets.yaml")


def path(key: str) -> Path:
    """Absolute path for a key in params.yaml -> paths (created if missing)."""
    p = ROOT / params()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
