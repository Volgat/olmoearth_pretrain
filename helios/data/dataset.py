"""Dataset module for helios."""

from typing import Literal, NamedTuple, cast

import numpy as np
import rioxarray
import xarray as xr
from einops import rearrange
from torch.utils.data import Dataset as PyTorchDataset
from upath import UPath

from helios.data.utils import (
    load_data_index,
    load_sentinel2_frequency_metadata,
    load_sentinel2_monthly_metadata,
)

# TODO: Move these to a .sources folder specific to each data source
# WARNING: TEMPORARY BANDS: We forgot to pull B9, B10 from the export
S2_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]


class DatasetOutput(NamedTuple):
    """A named tuple for storing the output of the dataset to the model (a single sample).

    Args:
        space_time_x: Input data that is space-time varying
        space_x: Input data that is space varying only
        time_x: Input data that is time varying only
        static_x: Input data that is static across space and time
        time_info: Input data that is time info namely all the time metadata for the time index
    """

    space_time_x: np.ndarray
    space_x: np.ndarray
    time_x: np.ndarray
    static_x: np.ndarray
    time_info: np.ndarray


# TODO: Adding a Dataset specific fingerprint is probably good for an evolving dataset
# TODO: We want to make what data sources and examples we use configuration drivend

# Quick and dirty interface for data sources
ALL_DATA_SOURCES = ["sentinel2_freq", "sentinel2_monthly"]

LOAD_DATA_SOURCE_METADATA_FUNCTIONS = {
    "sentinel2_freq": load_sentinel2_frequency_metadata,
    "sentinel2_monthly": load_sentinel2_monthly_metadata,
}

# Quick and dirty interface for data source variation types
DATA_SOURCE_VARIATION_TYPES = Literal[
    "space_time_varying", "time_varying_only", "space_varying_only", "static_only"
]

DATA_SOURCE_TO_VARIATION_TYPE = {
    "sentinel2_freq": "space_time_varying",
    "sentinel2_monthly": "space_time_varying",
}


# Expected types of Data Sources
# Space-TIme varying
# TIme varying only
# Space varying only
# Static only
# For a given location and or time we want to be able to coalesce the data sources
class HeliosDataset(PyTorchDataset):
    """Helios dataset."""

    def __init__(self, data_index_path: UPath | str, output_hw: int = 256):
        """Initialize the dataset."""
        self.data_index_path = UPath(data_index_path)
        # Using a df as initial ingest due to ease of inspection and manipulation,
        self.data_index_df = load_data_index(data_index_path)
        # Intersect available data sources with index column names
        self.data_sources = [
            source
            for source in ALL_DATA_SOURCES
            if source in self.data_index_df.columns
        ]
        print(self.data_sources)
        assert (
            len(self.data_sources) > 0
        ), "No data sources found in index, check naming of columns"
        print(self.data_index_df.head())
        self.example_ids = self.data_index_df["example_id"].to_numpy(dtype=str)
        self.output_hw = output_hw
        self.example_id_to_index_metadata_dict = self.data_index_df.set_index(
            "example_id"
        ).to_dict("index")
        self.root_dir = self.data_index_path.parent

        # Load metadata per data source so we can access quickly per data source per example
        self.data_source_metadata_dict = {}
        for data_source in self.data_sources:
            metadata_df = LOAD_DATA_SOURCE_METADATA_FUNCTIONS[data_source](
                self.get_path_to_data_source_metadata(data_source)
            )
            print(metadata_df.head())
            metadata_df.set_index(["example_id", "image_idx"], inplace=True, drop=True)
            # Structure of the metadata is {example_id, image_idx: {column: value}}
            example_id_to_data_source_metadata_dict = metadata_df.to_dict(
                orient="index"
            )

            self.data_source_metadata_dict[data_source] = (
                example_id_to_data_source_metadata_dict
            )

    def get_path_to_data_source_metadata(self, data_source: str) -> UPath:
        """Get the path to the data source metadata."""
        return self.root_dir / f"{data_source}.csv"

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.example_ids)

    def _tif_to_array(self, tif_path: UPath | str, data_source: str) -> np.ndarray:
        """Convert a tif file to an array.

        Args:
            tif_path: The path to the tif file.
            data_source: The data source string to load the correct datasource
        Returns:
            The array from the tif file.
        """
        if data_source == "sentinel2_freq":
            space_bands = S2_BANDS
        elif data_source == "sentinel2_monthly":
            space_bands = S2_BANDS
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        # We will need different ingestion logic for different data sources at this point

        variation_type = DATA_SOURCE_TO_VARIATION_TYPE[data_source]
        if variation_type == "space_time_varying":
            with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
                # [all_combined_bands, H, W]
                # all_combined_bands includes all dynamic-in-time bands
                # interleaved for all timesteps
                # followed by the static-in-time bands
                values = cast(np.ndarray, data.values)
                # lon = np.mean(cast(np.ndarray, data.x)).item()
                # lat = np.mean(cast(np.ndarray, data.y)).item()

            num_timesteps = values.shape[0] / len(space_bands)
            assert (
                num_timesteps % 1 == 0
            ), f"{tif_path} has incorrect number of channels {space_bands} \
                {values.shape[0]=} {len(space_bands)=}"
            dynamic_in_time_x = rearrange(
                values, "(t c) h w -> h w t c", c=len(space_bands), t=int(num_timesteps)
            )
            return dynamic_in_time_x
        else:
            raise NotImplementedError(f"Unknown variation type: {variation_type}")

    def _tif_to_array_with_checks(
        self, tif_path: UPath | str, data_source: str
    ) -> np.ndarray:
        """Load the tif file and return the array.

        Args:
            tif_path: The path to the tif file.
            data_source: The data source.

        Returns:
            The array from the tif file.
        """
        try:
            output = self._tif_to_array(tif_path, data_source)
            return output
        except Exception as e:
            print(f"Replacing tif {tif_path} due to {e}")
            raise e

    def _get_tif_path(self, data_source: str, example_id: str) -> UPath:
        return self.root_dir / data_source / f"{example_id}.tif"

    def __getitem__(self, index: int) -> DatasetOutput:
        """Get the item at the given index."""
        example_id = self.example_ids[index]
        index_metadata = self.example_id_to_index_metadata_dict[example_id]
        # check which data sources are available for this example
        data_sources_available_for_example = []
        for data_source in self.data_sources:
            if data_source in index_metadata.keys():
                if index_metadata[data_source] == "y":
                    data_sources_available_for_example.append(data_source)

        space_time_x = []
        space_x = []
        time_x = []
        static_x = []
        time_info = []
        for data_source in data_sources_available_for_example:
            tif_path = self._get_tif_path(data_source, example_id)
            data_source_variation_type = DATA_SOURCE_TO_VARIATION_TYPE[data_source]
            data_source_array = self._tif_to_array_with_checks(tif_path, data_source)

            # TODO: Confirm that this is what we want before we commit to a more optimal structure
            time_data_info = []
            for image_idx in range(data_source_array.shape[2]):
                time_data_info.append(
                    self.data_source_metadata_dict[data_source][
                        (example_id, image_idx)
                    ]["start_time"]
                )
            time_info.append(np.array(time_data_info))
            # grab the related time info for each index on the time axis
            if data_source_variation_type == "space_time_varying":
                space_time_x.append(data_source_array)
            elif data_source_variation_type == "time_varying_only":
                time_x.append(data_source_array)
            elif data_source_variation_type == "space_varying_only":
                space_x.append(data_source_array)
            elif data_source_variation_type == "static_only":
                static_x.append(data_source_array)

        # TODO: We will likely want to save thse numpy arrays locally and load directly those files
        # we will then need to decide how we will handle combining all the data sources together
        # What part of the dataset output the data belongs too depends on the type of the data source
        # concatenate on the time index dimensions are (h, w, t, c)
        space_time_x = np.concatenate(space_time_x, axis=2)
        ## SO FAR WE ARE NOT USING The below types of data sources
        space_x = np.empty([])
        time_x = np.empty([])
        static_x = np.empty([])
        time_info = np.concatenate(time_info, axis=0)
        return DatasetOutput(space_time_x, space_x, time_x, static_x, time_info)


if __name__ == "__main__":
    # TODO: Make this work for remote files likely want to use rslearn utils
    data_index_path = "gs://ai2-helios/data/20250113-sample-dataset-helios/index.csv"
    dataset = HeliosDataset(data_index_path)
    print(dataset[0])
