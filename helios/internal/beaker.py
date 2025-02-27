"""Launching Beaker experiments."""

from olmo_core.launch.beaker import BeakerLaunchConfig

HELIOS_DEFAULT_SETUP_STEPS = (
    'git clone "$REPO_URL" .',
    'git checkout "$GIT_REF"',
    "git submodule update --init --recursive",
    "conda shell.bash activate base",
    "pip install -e '.[all]'",
    "pip freeze",
)


class HeliosBeakerLaunchConfig(BeakerLaunchConfig):
    """Beaker launch config for Helios."""

    pass
