"""Tests for the load module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ta_forecasting.data.load import load_from_parquet, save_to_parquet


class TestSaveAndLoad:
    """Test Parquet save/load round-trip."""

    @pytest.fixture
    def sample_datasets(self) -> dict[str, pd.DataFrame]:
        """Create sample datasets."""
        return {
            "sheet_a": pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
            "sheet_b": pd.DataFrame({"x": [10.0, 20.0], "y": [30.0, 40.0]}),
        }

    def test_save_creates_files(self, tmp_path: Path, sample_datasets) -> None:
        """save_to_parquet creates .parquet files."""
        paths = save_to_parquet(sample_datasets, tmp_path)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            assert p.suffix == ".parquet"

    def test_roundtrip(self, tmp_path: Path, sample_datasets) -> None:
        """Data survives a save → load round-trip."""
        save_to_parquet(sample_datasets, tmp_path)
        loaded = load_from_parquet(tmp_path)
        assert set(loaded.keys()) == {"sheet_a", "sheet_b"}
        pd.testing.assert_frame_equal(
            loaded["sheet_a"].reset_index(drop=True),
            sample_datasets["sheet_a"],
        )

    def test_load_nonexistent_dir(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            load_from_parquet(tmp_path / "nonexistent")
