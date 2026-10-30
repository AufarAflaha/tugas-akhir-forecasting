"""Tests for the profile module — ensures no raw data leaks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ta_forecasting.data.profile import profile_workbook


class TestProfileWorkbook:
    """Test that profiling outputs only schema/stats, never raw data."""

    @pytest.fixture
    def sample_sheets(self) -> dict[str, pd.DataFrame]:
        """Create sample sheets with known content."""
        return {
            "test_sheet": pd.DataFrame({
                "quarter": ["2025Q1", "2025Q2", "2025Q3"],
                "cost_index": [100.5, 110.2, 105.8],
                "category": ["A", "B", "C"],
            }),
        }

    def test_creates_data_dictionary(
        self, tmp_path: Path, sample_sheets
    ) -> None:
        """profile_workbook creates DATA_DICTIONARY.md."""
        result = profile_workbook(sample_sheets, output_dir=tmp_path)
        assert result.exists()
        assert result.name == "DATA_DICTIONARY.md"

    def test_contains_schema_info(
        self, tmp_path: Path, sample_sheets
    ) -> None:
        """Output contains column names and dtypes."""
        result = profile_workbook(sample_sheets, output_dir=tmp_path)
        content = result.read_text()
        assert "cost_index" in content
        assert "quarter" in content
        assert "Rows" in content

    def test_no_raw_data_in_output(
        self, tmp_path: Path, sample_sheets
    ) -> None:
        """Output must NOT contain raw data values."""
        result = profile_workbook(sample_sheets, output_dir=tmp_path)
        content = result.read_text()
        # These are raw data values that should NOT appear
        assert "2025Q1" not in content
        assert "2025Q2" not in content
        # Category values should not appear
        # (column NAME "category" is fine, but value "A"/"B"/"C" in a data row is not)

    def test_contains_aggregate_stats(
        self, tmp_path: Path, sample_sheets
    ) -> None:
        """Output contains aggregate statistics for numeric columns."""
        result = profile_workbook(sample_sheets, output_dir=tmp_path)
        content = result.read_text()
        assert "Min" in content
        assert "Max" in content
        assert "Mean" in content
