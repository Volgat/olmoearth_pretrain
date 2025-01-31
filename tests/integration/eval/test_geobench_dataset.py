from pathlib import Path

import pytest
from einops import repeat

from helios.evals.datasets import GeobenchDataset
from helios.train.encoder import PatchEncoder


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
    model = PatchEncoder(in_channels=13, time_patch_size=1)

    # add a batch dimension
    s2_imagery = repeat(d[0]["s2"], "c t h w -> b c t h w", b=1)
    _ = model(s2_imagery.float())
