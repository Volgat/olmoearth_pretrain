"""Input/Output handlers for different data source types."""

from abc import ABC, abstractmethod

import numpy as np
import rasterio
from einops import rearrange
from upath import UPath

from helios.constants import NAIP_BANDS, S2_BANDS, WORLDCOVER_BANDS


class DataSourceReader(ABC):
    """Base class for data source readers.

    All readers should return data in (H, W, T, C) format where:
    - H, W are spatial dimensions
    - T is number of timesteps (1 for static data)
    - C is number of channels/bands
    """

    @classmethod
    @abstractmethod
    def load(cls, file_path: UPath | str) -> tuple[np.ndarray, int]:
        """Load data from file path.

        When Overiding this message defaults must be provided for extra kwargs
        """
        pass


class TiffReader(DataSourceReader):
    """Base reader for Tiff files."""

    @classmethod
    def load(
        cls, file_path: UPath | str, bands: list[str] = []
    ) -> tuple[np.ndarray, int]:
        """Load data from a Tiff file."""
        if not bands:
            # Including a default to satisfy mypy
            raise ValueError("Bands must be provided")
        with rasterio.open(file_path) as data:
            values = data.read()
        num_timesteps = values.shape[0] / len(bands)
        if not num_timesteps.is_integer():
            raise ValueError(
                f"{file_path} has incorrect number of channels {bands} "
                f"{values.shape[0]=} {len(bands)=}"
            )
        num_timesteps = int(num_timesteps)

        data_array = rearrange(
            values, "(t c) h w -> h w t c", c=len(bands), t=num_timesteps
        )

        return data_array, num_timesteps

    @classmethod
    def _check_bands(cls, bands: list[str], valid_bands: list[str]) -> None:
        """Check if the bands are valid."""
        if not all(band in valid_bands for band in bands):
            bands_not_in_valid = [band for band in bands if band not in valid_bands]
            raise ValueError(f"Invalid bands {bands_not_in_valid} for {cls.__name__}")


class GeoJSONReader(DataSourceReader):
    """Base reader for GeoJSON files."""

    @classmethod
    def load(cls, file_path: UPath | str) -> tuple[np.ndarray, int]:
        """Load data from a GeoJSON file."""
        # Convert GeoJSON to appropriate array format
        # Implementation depends on how we want to represent vector data
        return np.array([]), 1


class Sentinel2Reader(TiffReader):
    """Reader for Sentinel-2 data."""

    @classmethod
    def load(
        cls, file_path: UPath | str, bands: list[str] = S2_BANDS
    ) -> tuple[np.ndarray, int]:
        """Load Sentinel-2 data with specific band handling.

        Returns:
            Tuple of:
                - array of shape (H, W, T, C) where C is len(S2_BANDS)
                - number of timesteps T
        """
        cls._check_bands(bands, S2_BANDS)

        values, num_timesteps = super().load(file_path, bands=bands)

        return values, num_timesteps


class WorldCoverReader(TiffReader):
    """Reader for WorldCover data."""

    @classmethod
    def load(
        cls, file_path: UPath | str, bands: list[str] = WORLDCOVER_BANDS
    ) -> tuple[np.ndarray, int]:
        """Load WorldCover data.

        Returns:
            Tuple of:
                - array of shape (H, W, 1, C) containing land cover classes
                - always returns 1 timestep
        """
        cls._check_bands(bands, WORLDCOVER_BANDS)

        values, num_timesteps = super().load(file_path, bands=bands)
        if num_timesteps != 1:
            raise ValueError(
                f"WorldCover data must have 1 timestep, got {num_timesteps}"
            )

        return values, 1


class OpenStreetMapReader(GeoJSONReader):
    """Reader for OpenStreetMap data."""

    @classmethod
    def load(cls, file_path: UPath | str) -> tuple[np.ndarray, int]:
        """Load OpenStreetMap data."""
        return np.array([]), 1


class NAIPReader(TiffReader):
    """Reader for NAIP imagery."""

    @classmethod
    def load(
        cls, file_path: UPath | str, bands: list[str] = NAIP_BANDS
    ) -> tuple[np.ndarray, int]:
        """Load NAIP data.

        Returns:
            Tuple of:
                - array of shape (H, W, 1, C) containing  NAIP bands
                - always returns 1 timestep
        """
        cls._check_bands(bands, NAIP_BANDS)

        values, num_timesteps = super().load(file_path, bands=bands)
        if num_timesteps != 1:
            raise ValueError(f"NAIP data must have 1 timestep, got {num_timesteps}")

        return values, num_timesteps


class DataSourceLoaderRegistry:
    """Registry for data source readers."""

    _readers: dict[str, type[DataSourceReader]] = {}

    @classmethod
    def register(cls, source_name: str, reader: type[DataSourceReader]) -> None:
        """Register a reader for a data source."""
        cls._readers[source_name] = reader

    @classmethod
    def get_reader(cls, source_name: str) -> type[DataSourceReader]:
        """Get the reader for a data source."""
        if source_name not in cls._readers:
            raise ValueError(f"No reader registered for source: {source_name}")
        return cls._readers[source_name]


# Register all readers
DataSourceLoaderRegistry.register("sentinel2", Sentinel2Reader)
DataSourceLoaderRegistry.register("worldcover", WorldCoverReader)
DataSourceLoaderRegistry.register("openstreetmap", OpenStreetMapReader)
DataSourceLoaderRegistry.register("naip", NAIPReader)
