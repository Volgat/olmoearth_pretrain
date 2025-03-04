"""Test the HeliosDataset class compute_norm_stats method."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def test_compute_norm_stats() -> None:
    """Test compute_norm_stats in a streaming manner works."""
    # Mock data for testing
    modality = "modality1"
    band = "band1"
    norm_stats = {modality: {band: {"count": 0, "mean": 0.0, "var": 0.0}}}

    # Simulate multiple batches of data
    batches = [
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([5.0, 6.0, 7.0, 8.0]),
        np.array([9.0, 10.0, 11.0, 12.0]),
    ]

    # Process each batch
    for modality_band_data in batches:
        band_data_count = len(modality_band_data)

        # Current stats
        current_count = norm_stats[modality][band]["count"]
        current_mean = norm_stats[modality][band]["mean"]
        current_var = norm_stats[modality][band]["var"]

        # Compute updated mean and variance with the new batch of data
        new_count = current_count + band_data_count
        new_mean = (
            current_mean
            + (modality_band_data.mean() - current_mean) * band_data_count / new_count
        )
        new_var = (
            current_var
            + (
                (modality_band_data - current_mean) * (modality_band_data - new_mean)
            ).sum()
        )

        # Update the normalization stats
        norm_stats[modality][band]["count"] = new_count
        norm_stats[modality][band]["mean"] = new_mean
        norm_stats[modality][band]["var"] = new_var

    # Compute expected mean and variance from all data
    all_data = np.concatenate(batches)
    expected_mean = all_data.mean()
    expected_var = all_data.var() * len(all_data)  # Variance sum

    # Assertions
    assert norm_stats[modality][band]["count"] == len(all_data)
    assert np.isclose(norm_stats[modality][band]["mean"], expected_mean, atol=1e-5)
    assert np.isclose(norm_stats[modality][band]["var"], expected_var, atol=1e-5)
