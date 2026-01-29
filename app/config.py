from pathlib import Path
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else repo_root() / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg
