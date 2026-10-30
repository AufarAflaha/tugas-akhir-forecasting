"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

from ta_forecasting.config import Config, load_config, set_global_seed


class TestLoadConfig:
    """Test config loading from YAML."""

    def test_load_default_config(self, config: Config) -> None:
        """Default config.yaml loads without error."""
        assert config is not None
        assert isinstance(config, Config)

    def test_project_seed(self, config: Config) -> None:
        """Seed is set to 42."""
        assert config.project.seed == 42

    def test_last_actual_quarter(self, config: Config) -> None:
        """LAST_ACTUAL_QUARTER is 2026Q1."""
        assert config.project.last_actual_quarter == "2026Q1"

    def test_validation_ratios_sum(self, config: Config) -> None:
        """Train + val + test ratios sum to 1.0."""
        total = (
            config.validation.train_ratio
            + config.validation.val_ratio
            + config.validation.test_ratio
        )
        assert abs(total - 1.0) < 1e-9

    def test_data_raw_path(self, config: Config) -> None:
        """raw_path points to the xlsx file."""
        assert config.data.raw_path == Path("data/raw") / config.data.raw_file

    def test_metrics_primary(self, config: Config) -> None:
        """Primary metric is MAPE."""
        assert config.metrics.primary == "mape"

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing config."""
        import pytest

        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


class TestSetGlobalSeed:
    """Test seed setting."""

    def test_seed_sets_without_error(self) -> None:
        """set_global_seed should not raise."""
        set_global_seed(42)

    def test_reproducibility(self) -> None:
        """Same seed produces same random numbers."""
        import numpy as np

        set_global_seed(42)
        a = np.random.rand(5)
        set_global_seed(42)
        b = np.random.rand(5)
        assert (a == b).all()
