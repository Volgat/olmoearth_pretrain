"""Code for configuring and running Helios experiments."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from olmo_core.config import Config
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import ConfigSaverCallback, WandBCallback
from olmo_core.utils import get_default_device, seed_all

from helios.data.constants import ModalitySpec
from helios.data.dataloader import HeliosDataLoaderConfig
from helios.data.dataset import HeliosDatasetConfig, collate_helios
from helios.nn.latent_mim import LatentMIMConfig
from helios.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

# TODO: Make this more agnostic to the training setup
# TODO: Add support for different model configs
# TODO: Add support for different train module configs
# TODO: Add support for overrides


@dataclass
class CommonComponents(Config):
    """Any configurable items that are common to all experiments."""

    run_name: str
    save_folder: str
    supported_modalities: list[ModalitySpec]
    # callbacks: dict[str, Callback]


@dataclass
class HeliosExperimentConfig(Config):
    """Configuration for a Helios experiment."""

    run_name: str
    # launch: BeakerLaunchConfig # we should use this as well
    model: LatentMIMConfig  # TODO: make this agnostic to training setup
    dataset: HeliosDatasetConfig  # will likely be fixed for us
    data_loader: HeliosDataLoaderConfig  # will likely be fixed for us
    train_module: LatentMIMTrainModuleConfig  # we will want to support different train module model combinations
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(
    common: CommonComponents,
    model_config_builder: Callable[[CommonComponents], LatentMIMConfig],
    dataset_config_builder: Callable[[CommonComponents], HeliosDatasetConfig],
    dataloader_config_builder: Callable[[CommonComponents], HeliosDataLoaderConfig],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    train_module_config_builder: Callable[
        [CommonComponents], LatentMIMTrainModuleConfig
    ],
) -> HeliosExperimentConfig:
    """Build a Helios experiment configuration."""
    model_config = model_config_builder(common)
    dataset_config = dataset_config_builder(common)
    dataloader_config = dataloader_config_builder(common)
    trainer_config = trainer_config_builder(common)
    train_module_config = train_module_config_builder(common)
    config = HeliosExperimentConfig(
        run_name=common.run_name,
        model=model_config,
        dataset=dataset_config,
        data_loader=dataloader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    )
    return config


def train(config: HeliosExperimentConfig) -> None:
    """Train an experiment."""
    # Set RNG states on all devices. Also, done in prepare_training_environment
    seed_all(config.init_seed)

    # Build components.
    # TODO: Setup init device arg and allow the model to be inited on device of our choice rather than moved over
    model = config.model.build()
    device = get_default_device()
    model = model.to(device)
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(
        dataset, collator=collate_helios, dp_process_group=train_module.dp_process_group
    )
    trainer = config.trainer.build(train_module, data_loader)

    # Record the config to W&B/Comet and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    trainer.fit()


def run(config: HeliosExperimentConfig) -> None:
    """Run an experiment."""
    try:
        train(config)
    finally:
        teardown_training_environment()


def main(
    common_components_builder: Callable[[], CommonComponents],
    model_config_builder: Callable[[CommonComponents], LatentMIMConfig],
    dataset_config_builder: Callable[[CommonComponents], HeliosDatasetConfig],
    dataloader_config_builder: Callable[[CommonComponents], HeliosDataLoaderConfig],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    train_module_config_builder: Callable[
        [CommonComponents], LatentMIMTrainModuleConfig
    ],
) -> None:
    """Main entry point for Helios experiments."""
    common = common_components_builder()
    config = build_config(
        common=common,
        model_config_builder=model_config_builder,
        dataset_config_builder=dataset_config_builder,
        dataloader_config_builder=dataloader_config_builder,
        trainer_config_builder=trainer_config_builder,
        train_module_config_builder=train_module_config_builder,
    )

    prepare_training_environment(seed=config.init_seed)

    # Used for single GPU training
    # prepare_training_environment(seed=config.init_seed, backend=None)

    run(config)


# train

# a run method that does all this


# actually make this config
# set up the logic for each different experiment


# ask pete to make it more agnostic from language modeling
# Support logging all configs to WandB and to have an experiment config


# Extra features
# Parameter overrides
# debug modes
