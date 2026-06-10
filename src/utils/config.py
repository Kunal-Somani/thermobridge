"""Configuration loading and management via OmegaConf.

Supports YAML loading, CLI dotlist overrides, and dumping the fully resolved
config to a run directory for reproducibility (Rule 6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load a YAML config file and apply optional CLI dotlist overrides.

    Args:
        config_path: Path to the YAML configuration file.
        overrides: Optional list of dotlist overrides, e.g. ["training.lr=1e-3", "seed=123"].

    Returns:
        Fully merged and resolved OmegaConf DictConfig.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    base_cfg = OmegaConf.load(config_path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        base_cfg = OmegaConf.merge(base_cfg, override_cfg)

    # Make the config read-only after resolution to prevent accidental mutation
    OmegaConf.resolve(base_cfg)

    return base_cfg


def load_config_with_cli(config_path: str | Path) -> DictConfig:
    """Load config and merge any CLI arguments as dotlist overrides.

    Usage:
        python script.py training.lr=1e-3 seed=123

    Args:
        config_path: Path to the base YAML configuration file.

    Returns:
        Fully merged DictConfig with CLI overrides applied.
    """
    cli_overrides = sys.argv[1:]
    return load_config(config_path, overrides=cli_overrides if cli_overrides else None)


def save_resolved_config(cfg: DictConfig, output_dir: str | Path) -> Path:
    """Dump the fully resolved config to a YAML file in the output directory.

    This ensures every run has its exact configuration logged (Rule 6: reproducibility).

    Args:
        cfg: The resolved OmegaConf config to save.
        output_dir: Directory to write the resolved config file into.

    Returns:
        Path to the saved config file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "resolved_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            OmegaConf.to_container(cfg, resolve=True),
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    return config_path
