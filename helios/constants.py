"""Constants shared across the helios package."""

NAIP_BANDS = ["R", "G", "B", "IR"]

S2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]

WORLDCOVER_BANDS = ["B1"]


DATA_SOURCE_TO_VARIATION_TYPE = {
    "sentinel2": "space_time_varying",
}
