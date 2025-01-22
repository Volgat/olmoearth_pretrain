"""Test make_meta_summary script."""

import csv
import pathlib
from datetime import datetime, timezone

from upath import UPath

from helios.dataset_creation.const import METADATA_COLUMNS
from helios.dataset_creation.make_meta_summary import make_meta_summary
from helios.dataset_creation.util import get_modality_temp_meta_fname

# An example ID to use for testing with start_column placeholder.
TEST_EXAMPLE_ID = "EPSG:32610_10_{start_column}_456_2024-01-01T00:00:00+00:00"

TEST_MODALITY = "test"


def test_make_meta_summary(tmp_path: pathlib.Path) -> None:
    """Create two per-example metadata files and ensure they get concatenated."""
    example_ids = [
        TEST_EXAMPLE_ID.format(start_column=1),
        TEST_EXAMPLE_ID.format(start_column=2),
    ]
    image_times = [
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 2, 1, tzinfo=timezone.utc),
        datetime(2024, 3, 1, tzinfo=timezone.utc),
    ]

    helios_path = UPath(tmp_path)
    for example_id in example_ids:
        meta_fname = get_modality_temp_meta_fname(
            helios_path, TEST_MODALITY, example_id
        )
        meta_fname.parent.mkdir(parents=True, exist_ok=True)
        with meta_fname.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
            writer.writeheader()
            for image_idx, image_time in enumerate(image_times):
                writer.writerow(
                    dict(
                        example_id=example_id,
                        image_idx=image_idx,
                        start_time=image_time.isoformat(),
                        end_time=image_time.isoformat(),
                    )
                )

    make_meta_summary(helios_path, TEST_MODALITY)

    with (helios_path / f"{TEST_MODALITY}.csv").open() as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    assert len(csv_rows) == 6
    rows_by_example_and_idx = {
        (csv_row["example_id"], int(csv_row["image_idx"])): csv_row
        for csv_row in csv_rows
    }
    assert (
        rows_by_example_and_idx[(example_ids[0], 0)]["start_time"]
        == image_times[0].isoformat()
    )
    assert (
        rows_by_example_and_idx[(example_ids[0], 1)]["start_time"]
        == image_times[1].isoformat()
    )
    assert (
        rows_by_example_and_idx[(example_ids[0], 2)]["start_time"]
        == image_times[2].isoformat()
    )
    assert (
        rows_by_example_and_idx[(example_ids[1], 0)]["start_time"]
        == image_times[0].isoformat()
    )
    assert (
        rows_by_example_and_idx[(example_ids[1], 1)]["start_time"]
        == image_times[1].isoformat()
    )
    assert (
        rows_by_example_and_idx[(example_ids[1], 2)]["start_time"]
        == image_times[2].isoformat()
    )
