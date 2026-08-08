"""
Load configuration from file.
-----------------------------

src/config_loader.py
"""

import os
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """Hybrid configuration loader supporting .env + YAML + environment overrides."""

    def __init__(self, base_path: str = "config/settings.yaml"):
        # Load environment variables
        load_dotenv()
        self.env = os.getenv("HEDGEFORGE_ENV", "windows").lower()

        # Load base + environment-specific configs
        self.base = self._load_yaml(base_path)
        self.override = self._load_yaml(
            f"config/settings.{self.env}.yaml", optional=True
        )

        # Merge base + override
        self.config = self._merge(self.base, self.override)

        # Substitute env vars like ${DB_PASSWORD}
        self.config = self._substitute_env_vars(self.config)

    def _load_yaml(self, path: str, optional: bool = False) -> dict[str, Any]:
        if not os.path.exists(path):
            if optional:
                return {}
            raise FileNotFoundError(f"Required config file not found: {path}")
        with open(path, "r") as file:
            return yaml.safe_load(file) or {}

    def _merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge base and override YAML configs."""
        merged = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = self._merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _substitute_env_vars(self, data: Any) -> Any:
        """Recursively replace ${VAR} with os.getenv(VAR)."""
        if isinstance(data, dict):
            return {k: self._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(v) for v in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var = data[2:-1]
            return os.getenv(env_var, f"<missing:{env_var}>")
        return data

    def get(self, *keys: str, default: Any = None) -> Any:
        """Access nested config values safely."""
        node = self.config
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
        return node

    def as_dict(self) -> dict[str, Any]:
        return self.config
