"""Test fixtures and shared configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from ta_forecasting.config import Config, load_config

# Project root — two levels up from tests/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def config() -> Config:
    """Load the default project config."""
    return load_config(PROJECT_ROOT / "configs" / "config.yaml")


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT
