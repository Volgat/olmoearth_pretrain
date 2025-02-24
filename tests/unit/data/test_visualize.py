"""Unit tests for the Helios Dataset Visualization."""

from pathlib import Path

from helios.data.dataset import HeliosDataset
from helios.data.visualize import visualize_sample


def test_visualize_sample(
    prepare_samples_and_supported_modalities: tuple, tmp_path: Path
) -> None:
    """Test the visualize_sample function."""
    prepare_samples, supported_modalities = prepare_samples_and_supported_modalities
    samples = prepare_samples(tmp_path)
    dataset = HeliosDataset(
        samples=samples,
        supported_modalities=supported_modalities,
        tile_path=tmp_path,
    )
    for i in range(len(samples)):
        visualize_sample(dataset, i, tmp_path / "visualizations")
        assert (tmp_path / "visualizations" / f"sample_{i}.png").exists()
