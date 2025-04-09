"""This script is used to sweep different combinations of training modalities.

for latent_mim_debug.py.
"""

import subprocess  # nosec

from helios.data.constants import Modality, ModalitySpec

OLD_MODALITIES = [
    Modality.SENTINEL2_L2A,
    Modality.SENTINEL1,
    Modality.WORLDCOVER,
]

OLD_MODALITIES_SRTM = OLD_MODALITIES + [Modality.SRTM]

OLD_MODALITIES_SRTM_LANDSAT = OLD_MODALITIES_SRTM + [Modality.LANDSAT]

OLD_MODALITIES_SRTM_LANDSAT_OPENSTREETMAP = OLD_MODALITIES_SRTM_LANDSAT + [
    Modality.OPENSTREETMAP_RASTER
]

OLD_MODALITIES_LANDSAT = OLD_MODALITIES + [Modality.LANDSAT]

OLD_MODALITIES_OPENSTREETMAP = OLD_MODALITIES + [Modality.OPENSTREETMAP_RASTER]


# Combine all the combinations we want to test
MODALITY_COMBINATIONS = (
    OLD_MODALITIES,
    OLD_MODALITIES_SRTM,
    OLD_MODALITIES_SRTM_LANDSAT,
    OLD_MODALITIES_SRTM_LANDSAT_OPENSTREETMAP,
    OLD_MODALITIES_LANDSAT,
    OLD_MODALITIES_OPENSTREETMAP,
)

# Base command template
BASE_COMMAND = (
    "python3 scripts/new_dataset_base_debug/latent_mim_debug.py "
    "dry_run {run_name} ai2/jupiter-cirrascale-2 {modality_args}"
)


def format_training_modalities(modalities: list[ModalitySpec]) -> str:
    """Format the training modalities as a single comma-separated list."""
    # These modalities are used exactly as they are in latent_mim_debug.py
    # when defining TRAINING_MODALITIES
    formatted_modalities = []
    for m in modalities:
        # Convert modality name to lowercase to match format in latent_mim_debug.py
        modality_value = m.name.lower()
        formatted_modalities.append(f'"{modality_value}"')

    # Join with commas
    modality_list = ",".join(formatted_modalities)
    # Return as a command line argument
    return f"--common.training_modalities=[{modality_list}]"


# Iterate over all combinations of modalities
for modality_combo in MODALITY_COMBINATIONS:
    # Generate a descriptive name for the run
    modality_combo_name = "_".join([m.name.lower() for m in modality_combo])
    run_name = f"latentmim_modalities_{modality_combo_name}"

    # Format the modality arguments
    modality_args = format_training_modalities(modality_combo)

    # Construct full command
    command = BASE_COMMAND.format(
        run_name=run_name,
        modality_args=modality_args,
    )

    print(f"Launching: {command}")

    # Execute the command
    subprocess.run(command, shell=True, check=True)  # nosec
