"""Validated configuration with paths relative to code/business_entity_resolution."""

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "src/configs/default.json"


class ConfigError(ValueError):
    """Unreadable or invalid pipeline configuration."""


@dataclass(frozen=True)
class PathsConfig:
    train_data: Path
    test_data: Path
    artifacts: Path
    final_output: Path


@dataclass(frozen=True)
class PipelineConfig:
    paths: PathsConfig
    random_seed: int


def _resolve(value: str) -> Path:
    return (PROJECT_ROOT / Path(value).expanduser()).resolve()


def _keys(value: object, expected: set[str], section: str) -> None:
    if not isinstance(value, dict):
        raise ConfigError(f"{section} must be a JSON object")
    missing, unknown = expected - value.keys(), value.keys() - expected
    if missing or unknown:
        raise ConfigError(
            f"{section}: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}"
        )


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    """Load settings without creating directories or requiring input files.

    Relative config filenames and configured paths use PROJECT_ROOT, including
    paths in external config files. Absolute paths and ~ expansion are supported.
    Unknown keys are rejected to catch misspelled settings.
    """
    path = DEFAULT_CONFIG if config_path is None else _resolve(str(config_path))
    try:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigError(f"Cannot load configuration {path}: {exc}") from exc
    _keys(raw, {"paths", "random_seed"}, "configuration")
    _keys(raw["paths"], {"train_data", "test_data", "artifacts", "final_output"}, "paths")
    paths = {}
    for name, value in raw["paths"].items():
        if not isinstance(value, str) or not value.strip() or chr(0) in value:
            raise ConfigError(f"paths.{name} must be a nonempty path string")
        paths[name] = _resolve(value)
    # Protect original data and keep intermediate and final outputs separate.
    for left, right in (
        ("artifacts", "final_output"), ("artifacts", "train_data"),
        ("artifacts", "test_data"), ("final_output", "train_data"),
        ("final_output", "test_data"),
    ):
        a, b = paths[left], paths[right]
        if a == b or a in b.parents or b in a.parents:
            raise ConfigError(f"paths.{left} and paths.{right} must not overlap")
    seed = raw["random_seed"]
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ConfigError("random_seed must be an integer between 0 and 4294967295")
    return PipelineConfig(paths=PathsConfig(**paths), random_seed=seed)
