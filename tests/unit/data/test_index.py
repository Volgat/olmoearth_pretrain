"""Unit tests for parsing the dataset index."""

import pytest

from helios.data.index import DatasetIndexParser


@pytest.fixture
def sample_index_path() -> str:
    """Fixture providing path to test dataset index."""
    return "tests/fixtures/sample-dataset/index.csv"


def test_dataset_index(sample_index_path: str) -> None:
    """Test the dataset index."""
    index_parser = DatasetIndexParser(sample_index_path)
    assert len(index_parser.samples) == 6
    assert set(index_parser.samples[0].keys()) == {
        "data_source_paths",
        "data_source_metadata",
        "sample_metadata",
    }
