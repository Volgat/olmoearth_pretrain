"""Unit tests for HeliosSample."""

import torch

from helios.data.dataset import HeliosSample


def test_all_attrs_have_bands() -> None:
    """Test all attributes are described in attribute_to_bands."""
    for attribute_name in HeliosSample._fields:
        _ = HeliosSample.num_bands(attribute_name)


def test_subsetting() -> None:
    """Test subsetting works."""
    (
        b,
        h,
        w,
        t,
    ) = 1, 16, 16, 3
    sample = HeliosSample(
        sentinel2=torch.ones((b, h, w, t, HeliosSample.num_bands("sentinel2"))),
        timestamps=torch.ones((b, t, HeliosSample.num_bands("timestamps"))),
    )
    _ = sample.subset(
        patch_size=4, max_tokens_per_instance=1500, hw_to_sample=[1, 2, 3, 4]
    )
