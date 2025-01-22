"""Unit tests for the data source IO module."""

from unittest.mock import patch

import numpy as np
import pytest
from upath import UPath

from helios.constants import S2_BANDS
from helios.data.data_source_io import (
    DataSourceLoaderRegistry,
    NAIPReader,
    OpenStreetMapReader,
    Sentinel2Reader,
    WorldCoverReader,
)


@pytest.fixture
def mock_tiff_file(tmp_path: str) -> UPath:
    """Create a mock TIFF file for testing."""
    # Mock file path that our readers will use
    return UPath(tmp_path) / "test.tif"


class TestSentinel2Reader:
    """Tests for Sentinel2Reader."""

    def test_load_data(self, mock_tiff_file: UPath) -> None:
        """Test Sentinel2Reader loads and reshapes data correctly."""
        num_timesteps = 2

        # Create mock data with shape (bands * timesteps, height, width)
        mock_data = np.random.rand(len(S2_BANDS) * num_timesteps, 64, 64).astype(
            np.float32
        )

        with patch("rasterio.open") as mock_rasterio:
            # Configure mock to return our test data
            mock_rasterio.return_value.__enter__.return_value.read.return_value = (
                mock_data
            )

            # Test loading
            data_array, timesteps = Sentinel2Reader.load(mock_tiff_file)

        # Verify shape and properties
        assert timesteps == num_timesteps
        assert data_array.shape == (64, 64, num_timesteps, len(S2_BANDS))
        assert isinstance(data_array, np.ndarray)

    def test_invalid_bands(self, mock_tiff_file: UPath) -> None:
        """Test Sentinel2Reader raises error with invalid number of bands."""
        # Create mock data with invalid number of bands
        mock_data = np.random.rand(7, 64, 64)  # Invalid number of bands

        with patch("rasterio.open") as mock_rasterio:
            mock_rasterio.return_value.__enter__.return_value.read.return_value = (
                mock_data
            )
            with pytest.raises(ValueError):
                Sentinel2Reader.load(mock_tiff_file)


class TestWorldCoverReader:
    """Tests for WorldCoverReader."""

    def test_load_data(self, mock_tiff_file: UPath) -> None:
        """Test WorldCoverReader loads and reshapes data correctly."""
        # Create mock data with shape (1, height, width)
        mock_data = np.random.randint(0, 10, (1, 64, 64)).astype(np.float32)

        with patch("rasterio.open") as mock_rasterio:
            mock_rasterio.return_value.__enter__.return_value.read.return_value = (
                mock_data
            )
            data_array, timesteps = WorldCoverReader.load(mock_tiff_file)

        assert timesteps == 1
        assert data_array.shape == (64, 64, 1, 1)
        assert isinstance(data_array, np.ndarray)


class TestOpenStreetMapReader:
    """Tests for OpenStreetMapReader."""

    def test_load_data(self) -> None:
        """Test OpenStreetMapReader creates correct placeholder."""
        data_array, timesteps = OpenStreetMapReader.load("dummy_path")

        assert timesteps == 1
        assert data_array.shape == (256, 256, 1, 3)
        assert data_array.dtype == np.float32
        assert np.all(data_array == 0)  # Should be initialized with zeros


class TestNAIPReader:
    """Tests for NAIPReader."""

    def test_load_data(self, mock_tiff_file: UPath) -> None:
        """Test NAIPReader loads and reshapes data correctly."""
        # Create mock data with shape (4, height, width) for RGBI bands
        mock_data = np.random.rand(4, 64, 64).astype(np.float32)

        with patch("rasterio.open") as mock_rasterio:
            mock_rasterio.return_value.__enter__.return_value.read.return_value = (
                mock_data
            )
            data_array, timesteps = NAIPReader.load(mock_tiff_file)

        assert timesteps == 1
        assert data_array.shape == (64, 64, 1, 4)
        assert isinstance(data_array, np.ndarray)


class TestDataSourceLoaderRegistry:
    """Tests for DataSourceLoaderRegistry."""

    def test_registration(self) -> None:
        """Test DataSourceLoaderRegistry registration and retrieval."""
        # Test getting registered readers
        assert DataSourceLoaderRegistry.get_reader("sentinel2") == Sentinel2Reader
        assert DataSourceLoaderRegistry.get_reader("worldcover") == WorldCoverReader
        assert (
            DataSourceLoaderRegistry.get_reader("openstreetmap") == OpenStreetMapReader
        )
        assert DataSourceLoaderRegistry.get_reader("naip") == NAIPReader

        # Test error on unknown reader
        with pytest.raises(ValueError, match="No reader registered for source"):
            DataSourceLoaderRegistry.get_reader("unknown_source")
