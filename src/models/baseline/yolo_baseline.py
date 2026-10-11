"""YOLOv8 (ultralytics) baseline — trained/evaluated on KACHARA as-published,
no tiling, no scale-fusion head, no severity scoring. Comparison point only;
see docs/contribution-notes.md for what this excludes relative to the
contribution pipeline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ultralytics import YOLO


def _make_time_budget_callback(budget_seconds: float, logger: logging.Logger | None = None):
    """Stops training after the epoch currently finishing if starting another
    would likely exceed budget_seconds, based on the mean per-epoch time
    observed so far. Must be registered on "on_fit_epoch_end" specifically --
    that event fires after validation AND checkpoint saving for the epoch are
    already done (see BaseTrainer._do_train), so `trainer.stop = True` here
    only ever takes effect between epochs, never mid-epoch. Ultralytics' own
    `time=` arg is NOT used for this because it's also checked mid-batch,
    which could cut off a run before that epoch's checkpoint is saved.

    BaseTrainer.final_eval() deliberately re-fires "on_fit_epoch_end" once
    more after training ends (with trainer.epoch bumped by 1, to log
    best-model metrics at a synthetic final step) -- this happens on every
    run, not just early-stopped ones. The `already_stopped` guard below
    makes sure that bonus call never re-triggers or re-logs this callback.
    """
    already_stopped = False

    def _callback(trainer) -> None:
        nonlocal already_stopped
        if already_stopped:
            return
        elapsed = time.time() - trainer.train_time_start
        epochs_completed = trainer.epoch + 1
        mean_epoch_time = elapsed / epochs_completed
        if elapsed + mean_epoch_time > budget_seconds:
            trainer.stop = True
            already_stopped = True
            message = (
                f"Soft time budget reached: {elapsed:.1f}s elapsed after {epochs_completed} "
                f"epoch(s) (mean {mean_epoch_time:.1f}s/epoch); another epoch would likely "
                f"exceed the {budget_seconds:.0f}s budget. Stopping after epoch "
                f"{epochs_completed} (already checkpointed) instead of starting the next one."
            )
            if logger is not None:
                logger.warning(message)
            else:
                print(message)

    return _callback


class YOLOv8Baseline:
    def __init__(self, model_variant: str = "yolov8n.pt", device: str | None = None):
        self.model = YOLO(model_variant)
        self.device = device

    def train(
        self,
        data_yaml: str | Path,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        project: str | Path = "experiments/checkpoints",
        name: str = "yolov8_baseline",
        time_budget_seconds: float | None = None,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        if time_budget_seconds is not None:
            self.model.add_callback(
                "on_fit_epoch_end", _make_time_budget_callback(time_budget_seconds, logger)
            )

        # ultralytics silently nests relative `project` paths under its own
        # runs/<task>/ root (see ultralytics.cfg.get_save_dir) -- resolve to
        # absolute so checkpoints land exactly where the config says.
        return self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=str(Path(project).resolve()),
            name=name,
            **kwargs,
        )

    def evaluate(self, data_yaml: str | Path, split: str = "test", imgsz: int = 640, **kwargs):
        return self.model.val(
            data=str(data_yaml), split=split, imgsz=imgsz, device=self.device, **kwargs
        )

    def predict(self, source, imgsz: int = 640, conf: float = 0.25, **kwargs):
        return self.model.predict(
            source=source, imgsz=imgsz, conf=conf, device=self.device, **kwargs
        )

    @classmethod
    def from_checkpoint(cls, weights_path: str | Path, device: str | None = None) -> "YOLOv8Baseline":
        instance = cls.__new__(cls)
        instance.model = YOLO(str(weights_path))
        instance.device = device
        return instance
