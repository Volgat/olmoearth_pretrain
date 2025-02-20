"""Downstream evaluator callback."""

import logging
import time
from dataclasses import dataclass, field

import torch
from olmo_core.eval.evaluator import Evaluator
from olmo_core.train.callbacks.callback import Callback, CallbackConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from torch.utils.data import DataLoader
from upath import UPath

from helios.evals.datasets import GeobenchDataset
from helios.evals.embeddings import get_embeddings
from helios.evals.knn import run_knn

logger = logging.getLogger(__name__)


# Default metric name for classification
METRIC_NAME = "f1"
NAME_PREFIX = "Geobench"
GEOBENCH_DIR = UPath("/weka/dfive-default/presto-geobench/dataset/geobench")


class DownstreamEvaluator:
    """Evaluator for downstream tasks."""

    def __init__(
        self,
        name: str,
        task: str,
        trainer: Trainer,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the downstream evaluator."""
        self.name = name
        self.task = task
        self.trainer = trainer
        self.device = device

    def val(self) -> float:
        """Validate the model on the downstream task."""
        train_ds = GeobenchDataset(GEOBENCH_DIR, self.task, "train", "default")
        train_loader = DataLoader(train_ds, collate_fn=GeobenchDataset.collate_fn)
        val_loader = DataLoader(
            GeobenchDataset(GEOBENCH_DIR, self.task, "valid", "default"),
            collate_fn=GeobenchDataset.collate_fn,
        )
        train_embeddings, train_labels = get_embeddings(
            data_loader=train_loader,
            model=self.trainer.train_module.model.target_encoder,
            patch_size=self.trainer.train_module.model.encoder.max_patch_size,
        )
        val_embeddings, test_labels = get_embeddings(
            data_loader=val_loader,
            model=self.trainer.train_module.model.target_encoder,
            patch_size=self.trainer.train_module.model.encoder.max_patch_size,
        )
        val_result = run_knn(
            eval_type="KNN-20",
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=val_embeddings,
            test_labels=test_labels,
            num_classes=train_ds.num_classes,
            is_multilabel=train_ds.is_multilabel,
            device=self.device,
        )
        logger.info(
            f"Downstream evaluator {self.name} {METRIC_NAME} score: {val_result}"
        )
        return val_result


@dataclass
class DownstreamEvaluatorCallback(Callback):
    """Runs in-loop evaluations periodically during training."""

    # The evaluators to run
    evaluators: list[DownstreamEvaluator] = field(default_factory=list)

    # The interval (in steps) with which to run the evaluators
    eval_interval: int = 10

    # The duration to run each evaluator for
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(10))

    def post_step(self) -> None:
        """Run the evaluators."""
        if self.step <= 1 or self.step % self.eval_interval != 0:
            return

        # dp_world_size = get_world_size(self.trainer.dp_process_group)

        for evaluator in self.evaluators:
            logger.info(f"Running {evaluator.name} evals...")
            start_time = time.monotonic()
            # Run validation
            val_result = evaluator.val()
            self.trainer.record_metric(
                f"eval/{evaluator.name}/{METRIC_NAME}", val_result
            )
            logger.info(
                f"Finished {evaluator.name} evals in {time.monotonic() - start_time:.1f} seconds. {METRIC_NAME}: {val_result}"
            )


@dataclass
class DownstreamEvaluatorCallbackConfig(CallbackConfig):
    """Config for the downstream evaluator callback."""

    tasks: list[str]
    eval_interval: int = 10
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(10))
    enabled: bool = True

    def build(self, trainer: "Trainer") -> Callback | None:
        """Build the downstream evaluator callback."""
        if not self.enabled:
            return None

        evaluators: list[Evaluator] = []
        for task in self.tasks:
            evaluators.append(
                DownstreamEvaluator(
                    name=f"{NAME_PREFIX}-{task}",
                    task=task,
                    trainer=trainer,
                    device=trainer.device,
                )
            )

        return DownstreamEvaluatorCallback(
            evaluators=evaluators,
            eval_interval=self.eval_interval,
            eval_duration=self.eval_duration,
        )
