"""Post-process ingested WorldCereal data into the Helios dataset."""

import argparse
import csv
import multiprocessing
from datetime import datetime, timezone

import numpy as np
import tqdm
from rslearn.dataset import Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from helios.data.constants import Modality, TimeSpan
from helios.dataset.utils import get_modality_fname

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata

START_TIME = datetime(2021, 1, 1, tzinfo=timezone.utc)
END_TIME = datetime(2022, 1, 1, tzinfo=timezone.utc)


def _fill_nones_with_zeros(ndarrays: list[np.ndarray | None]) -> np.ndarray | None:
    filler = None
    for x in ndarrays:
        if x is not None:
            filler = np.zeros_like(x)
            break
    if filler is None:
        return None

    return_list = []
    for x in ndarrays:
        if x is not None:
            return_list.append(x)
        else:
            return_list.append(filler.copy())
    return np.concatenate(return_list, axis=0)


def scale_confidences(x: np.ndarray) -> np.ndarray:
    """From eq.1 of the worldcereal paper.

    As a complementary product of the binary prediction,
    the models also provide binary class probabilities which we
    used to assess the pixel-based model's confidence in its prediction.
    Unconfident model predictions are characterized by binary probabilities
    close to 0.5, while confident model predictions are close to 0 or 1.
    Therefore, we defined model confidence as a value between 0 and 100, computed
    using Eq. (1):

    confidence = ((probability - 0.5) / 0.5) * 100

    This function reverses eq. (1) to return to a probability.
    Probability is between 0.5 and 1, where 0.5 == not confident, and 1 == confident.
    """
    return ((x / 100) / 2) + 0.5


def to_probabilities(binary: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Take the binary (1, 0) and confidence (< 0.5) values and return probabilities."""
    confidences = scale_confidences(confidence)
    # the resampling may have made the values non-binary, so lets binarize them again.
    binary = binary >= 0.5
    # since the output of scale_confidences, c, is between 0.5 and 1, I want
    # (1 - c) where binary == 0 and (c) where binary == 1
    return (binary * confidences) + ((1 - binary) * (1 - confidences))


def convert_worldcereal(window_path: UPath, helios_path: UPath) -> None:
    """Add WorldCereal data for this window to the Helios dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: Helios dataset path to write to.
    """
    ndarrays: list[np.ndarray | None] = []
    assert len(Modality.WORLDCEREAL.band_sets) == 1
    band_set = Modality.WORLDCEREAL.band_sets[0]
    window = Window.load(window_path)
    window_metadata = get_window_metadata(window)
    for band in band_set.bands:
        confidence_layer = f"{band}-confidence"
        classification_layer = f"{band}-classification"
        both_done = [
            window.is_layer_completed(confidence_layer),
            window.is_layer_completed(classification_layer),
        ]
        if sum(both_done) == 0:
            ndarrays.append(None)
            continue
        elif sum(both_done) == 1:
            raise RuntimeError(
                "Expected both the confidence and classification layers, or neither. "
                f"Got one for {window_path}, {band}"
            )

        binary_dir = window.get_raster_dir(classification_layer, [classification_layer])
        confidence_dir = window.get_raster_dir(confidence_layer, [confidence_layer])

        ndarrays.append(
            to_probabilities(
                binary=GEOTIFF_RASTER_FORMAT.decode_raster(
                    path=binary_dir, projection=window.projection, bounds=window.bounds
                ),
                confidence=GEOTIFF_RASTER_FORMAT.decode_raster(
                    path=confidence_dir,
                    projection=window.projection,
                    bounds=window.bounds,
                ),
            )
        )

    assert len(ndarrays) == len(
        band_set.bands
    ), f"Expected {len(band_set.bands)} arrays, got {len(ndarrays)}"
    concatenated_arrays = _fill_nones_with_zeros(ndarrays)

    if concatenated_arrays is None:
        return None

    assert concatenated_arrays.min() >= 0
    assert concatenated_arrays.max() <= 1

    dst_fname = get_modality_fname(
        helios_path,
        Modality.WORLDCEREAL,
        TimeSpan.STATIC,
        window_metadata,
        band_set.get_resolution(),
        "tif",
    )
    GEOTIFF_RASTER_FORMAT.encode_raster(
        path=dst_fname.parent,
        projection=window.projection,
        bounds=window.bounds,
        array=concatenated_arrays,
        fname=dst_fname.name,
    )
    metadata_fname = get_modality_temp_meta_fname(
        helios_path, Modality.WORLDCEREAL, TimeSpan.STATIC, window.name
    )
    metadata_fname.parent.mkdir(parents=True, exist_ok=True)
    with metadata_fname.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerow(
            dict(
                crs=window_metadata.crs,
                col=window_metadata.col,
                row=window_metadata.row,
                tile_time=window_metadata.time.isoformat(),
                image_idx="0",
                start_time=START_TIME.isoformat(),
                end_time=END_TIME.isoformat(),
            )
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Post-process Helios data",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Source rslearn dataset path",
        required=True,
    )
    parser.add_argument(
        "--helios_path",
        type=str,
        help="Destination Helios dataset path",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to use",
        default=32,
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    helios_path = UPath(args.helios_path)

    metadata_fnames = ds_path.glob("windows/*/*/metadata.json")
    jobs = []
    for metadata_fname in metadata_fnames:
        jobs.append(
            dict(
                window_path=metadata_fname.parent,
                helios_path=helios_path,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_worldcereal, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
