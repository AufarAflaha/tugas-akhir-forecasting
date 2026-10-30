"""Tests for the extract module."""

from __future__ import annotations

from pathlib import Path

import pytest

from ta_forecasting.data.extract import extract_workbook


class TestExtractWorkbook:
    """Test data extraction from the workbook."""

    def test_workbook_exists(self, config) -> None:
        """The raw .xlsx file exists at the configured path."""
        assert config.data.raw_path.exists(), (
            f"Workbook not found at {config.data.raw_path}"
        )

    def test_extract_returns_dict(self, config) -> None:
        """extract_workbook returns a dict of DataFrames."""
        result = extract_workbook(config.data.raw_path)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_extract_sheets_have_rows(self, config) -> None:
        """Each extracted sheet has at least one row."""
        result = extract_workbook(config.data.raw_path)
        for name, df in result.items():
            assert len(df) > 0, f"Sheet '{name}' is empty"

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            extract_workbook(tmp_path / "nonexistent.xlsx")
