"""Tests for the transform module."""

from __future__ import annotations

import pandas as pd
import pytest

from ta_forecasting.data.transform import filter_after_last_actual


class TestFilterAfterLastActual:
    """Test temporal filtering logic."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame with quarterly data."""
        return pd.DataFrame({
            "quarter": ["2025Q1", "2025Q2", "2025Q3", "2025Q4",
                        "2026Q1", "2026Q2", "2026Q3", "2026Q4"],
            "value": [100, 110, 120, 130, 140, 150, 160, 170],
        })

    def test_filter_removes_projections(self, sample_df: pd.DataFrame) -> None:
        """Rows after LAST_ACTUAL_QUARTER should be removed."""
        result = filter_after_last_actual(sample_df, "quarter", "2026Q1")
        assert len(result) == 5  # 2025Q1..2026Q1

    def test_filter_keeps_last_actual(self, sample_df: pd.DataFrame) -> None:
        """The LAST_ACTUAL_QUARTER itself should be kept."""
        result = filter_after_last_actual(sample_df, "quarter", "2026Q1")
        quarters = result["quarter"].tolist()
        assert "2026Q1" in quarters

    def test_filter_excludes_future(self, sample_df: pd.DataFrame) -> None:
        """Quarters after cutoff should NOT be present."""
        result = filter_after_last_actual(sample_df, "quarter", "2026Q1")
        quarters = result["quarter"].tolist()
        assert "2026Q2" not in quarters
        assert "2026Q3" not in quarters
        assert "2026Q4" not in quarters

    def test_filter_all_before_cutoff(self) -> None:
        """If all data is before cutoff, nothing is removed."""
        df = pd.DataFrame({
            "quarter": ["2020Q1", "2020Q2"],
            "value": [1, 2],
        })
        result = filter_after_last_actual(df, "quarter", "2026Q1")
        assert len(result) == 2

    def test_filter_all_after_cutoff(self) -> None:
        """If all data is after cutoff, everything is removed."""
        df = pd.DataFrame({
            "quarter": ["2027Q1", "2027Q2"],
            "value": [1, 2],
        })
        result = filter_after_last_actual(df, "quarter", "2026Q1")
        assert len(result) == 0
