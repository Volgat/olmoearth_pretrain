from pathlib import Path

import pytest

from helios.evals.datasets import GeobenchDataset


@pytest.fixture
def geobench_dir() -> Path:
    """Fixture providing path to test dataset index."""
    return Path("tests/fixtures/sample_geobench")


def test_geobench_dataset(geobench_dir):
    d = GeobenchDataset(
        dataset="m-eurosat",
        geobench_dir=geobench_dir,
        split="train",
        partition="0.01x_train",
    )
    sample = d[0]
    assert "s2" in sample
    assert "target" in sample
    assert sample["s2"].shape == (13, 1, 64, 64)
