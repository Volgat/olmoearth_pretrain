import json
from pathlib import Path
from types import MethodType

import geobench
import numpy as np
import torch.multiprocessing
from einops import repeat
from torch.utils.data import Dataset

torch.multiprocessing.set_sharing_strategy("file_system")


S2_BAND_NAMES = [
    "01 - Coastal aerosol",
    "02 - Blue",
    "03 - Green",
    "04 - Red",
    "05 - Vegetation Red Edge",
    "06 - Vegetation Red Edge",
    "07 - Vegetation Red Edge",
    "08 - NIR",
    "08A - Vegetation Red Edge",
    "09 - Water vapour",
    "10 - SWIR - Cirrus",
    "11 - SWIR",
    "12 - SWIR",
]


class GeobenchDataset(Dataset):
    def __init__(
        self,
        geobench_dir: Path,
        dataset: str,
        split: str,
        partition: str,
        norm_method: str = "norm_no_clip",
    ):
        with (Path(__file__).parents[0] / f"{dataset}.json").open("r") as f:
            config = json.load(f)

        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Excected split to be in ['train', 'valid', 'test'], got {split}"
            )
        assert split in ["train", "valid", "test"]

        self.split = split
        self.config = config
        self.partition = partition

        for task in geobench.task_iterator(
            benchmark_name=self.config["benchmark_name"],
            benchmark_dir=geobench_dir / self.config["benchmark_name"],
        ):
            if task.dataset_name == self.config["dataset_name"]:
                break

        # hack: https://github.com/ServiceNow/geo-bench/issues/22
        task.get_dataset_dir = MethodType(
            lambda self: geobench_dir / config["benchmark_name"] / self.dataset_name,
            task,
        )

        self.dataset = task.get_dataset(split=self.split, partition_name=self.partition)
        original_band_names = [
            self.dataset[0].bands[i].band_info.name
            for i in range(len(self.dataset[0].bands))
        ]

        self.band_names = list(self.config["band_info"].keys())
        self.band_indices = [
            original_band_names.index(band_name) for band_name in self.band_names
        ]
        imputed_band_info = self.impute_normalization_stats(
            self.config["band_info"], self.config["imputes"]
        )
        self.mean, self.std = self.get_norm_stats(imputed_band_info)
        self.active_indices = range(int(len(self.dataset)))
        self.norm_method = norm_method

    @staticmethod
    def get_norm_stats(imputed_band_info: list[float]):
        means = []
        stds = []
        for band_name in S2_BAND_NAMES:
            assert band_name in imputed_band_info, f"{band_name} not found in band_info"
            means.append(imputed_band_info[band_name]["mean"])
            stds.append(imputed_band_info[band_name]["std"])
        return np.array(means), np.array(stds)

    @staticmethod
    def impute_normalization_stats(band_info: list[float], imputes: list[str]):
        # band_info is a dictionary with band names as keys and statistics (mean / std) as values
        if not imputes:
            return band_info

        names_list = list(band_info.keys())
        new_band_info = {}
        for band_name in S2_BAND_NAMES:
            new_band_info[band_name] = {}
            if band_name in names_list:
                # we have the band, so use it
                new_band_info[band_name] = band_info[band_name]
            else:
                # we don't have the band, so impute it
                for impute in imputes:
                    src, tgt = impute
                    if tgt == band_name:
                        # we have a match!
                        new_band_info[band_name] = band_info[src]
                        break

        return new_band_info

    @staticmethod
    def impute_bands(
        image_list: list[np.ndarray], names_list: list[str], imputes: list[str]
    ):
        # image_list should be one np.array per band, stored in a list
        # image_list and names_list should be ordered consistently!
        if not imputes:
            return image_list

        # create a new image list by looping through and imputing where necessary
        new_image_list = []
        for band_name in S2_BAND_NAMES:
            if band_name in names_list:
                # we have the band, so append it
                band_idx = names_list.index(band_name)
                new_image_list.append(image_list[band_idx])
            else:
                # we don't have the band, so impute it
                for impute in imputes:
                    src, tgt = impute
                    if tgt == band_name:
                        # we have a match!
                        band_idx = names_list.index(src)
                        new_image_list.append(image_list[band_idx])
                        break
        return new_image_list

    @staticmethod
    def normalize_bands(
        image: np.ndarray, means: np.array, stds: np.array, method: str = "norm_no_clip"
    ):
        original_dtype = image.dtype

        if method == "standardize":
            image = (image - means) / stds
        else:
            min_value = means - stds
            max_value = means + stds
            image = (image - min_value) / (max_value - min_value)

            if method == "norm_yes_clip":
                image = np.clip(image, 0, 1)
            elif method == "norm_yes_clip_int":
                # same as clipping between 0 and 1 but rounds to the nearest 1/255
                image = image * 255  # scale
                image = np.clip(image, 0, 255).astype(
                    np.uint8
                )  # convert to 8-bit integers
                image = (
                    image.astype(original_dtype) / 255
                )  # back to original_dtype between 0 and 1
            elif method == "norm_no_clip":
                pass
            else:
                raise ValueError(
                    f"norm type must be norm_yes_clip, norm_yes_clip_int, norm_no_clip, or standardize, not {method}"
                )
        return image

    def __getitem__(self, idx):
        label = self.dataset[idx].label

        x = []
        for band_idx in self.band_indices:
            x.append(self.dataset[idx].bands[band_idx].data)

        x = self.impute_bands(x, self.band_names, self.config["imputes"])

        x = np.stack(x, axis=2)  # (h, w, 13)
        assert (
            x.shape[-1] == 13
        ), f"All datasets must have 13 channels, not {x.shape[-1]}"
        if self.config["dataset_name"] == "m-so2sat":
            x = x * 10_000

        x = torch.tensor(self.normalize_bands(x, self.mean, self.std, self.norm_method))

        # check if label is an object or a number
        if not (isinstance(label, int) or isinstance(label, list)):
            label = label.data
            # label is a memoryview object, convert it to a list, and then to a numpy array
            label = np.array(list(label))

        target = torch.tensor(label, dtype=torch.long)

        return {"s2": repeat(x, "h w c -> c t h w", t=1), "target": target}

    def __len__(self):
        return len(self.dataset)
