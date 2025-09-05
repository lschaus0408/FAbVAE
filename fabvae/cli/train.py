"""
--------------------    FAbVAE     ----------------------
FAbVAE is a ...
Author: Lucas Schaus
------------------- AbVAE Training ----------------------
AbVAE Training Script. Contains details on how the models
are trained as well as various schedulers for training
parameters.
"""

from __future__ import annotations

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)


import argparse
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from fabvae.data.load_sequences import (
    BaseDataLoader,
    ProteinSequenceLoader,
    one_hot_tokenizer,
)
from fabvae.models.base_model.base_model import FAbVAEBase
from fabvae.models.base_model.vae import AbVAEBase

### --------------------------------------------------------###
#                       Training Utils                        #
### --------------------------------------------------------###


class VAEModule(pl.LightningModule):
    """
    ## Lightning Wrapper Around FAbVAE Modules
    Expects the model to expose:
        \telbo(batch) -> (loss, recon, kl, beta)\n
        \tmutable attribute set by Scheduler used during elbo
    """

    def __init__(
        self,
        model: FAbVAEBase,
        learning_rate: float = 3e-4,
        lr_optimizer_factor: float = 0.5,
        lr_optimizer_patience: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.learning_rate = learning_rate
        self.lr_optimizer_factor = lr_optimizer_factor
        self.lr_optimizer_patience = lr_optimizer_patience

    def forward(  # pylint: disable=arguments-differ
        self, tensor_input: torch.Tensor
    ) -> torch.Tensor:
        """
        ## Forward Pass
        """
        return self.model(tensor_input)

    def training_step(self, batch: torch.Tensor, _: int):  # pylint: disable=arguments-differ
        """
        ## One Training Step of Model
        """
        total_loss, reconstruction_loss, kl_loss, beta = self.model.elbo(batch)
        # Log the average losses
        self.log("train/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/recon", reconstruction_loss, on_step=False, on_epoch=True)
        self.log("train/kl", kl_loss, on_step=False, on_epoch=True)
        self.log("train/beta", beta, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch: torch.Tensor, _: int):  # pylint: disable=arguments-differ
        """
        ## One Validation Step of Model
        """
        total_loss, reconstruction_loss, kl_loss, beta = self.model.elbo(batch)
        self.log("val/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/recon", reconstruction_loss, on_step=False, on_epoch=True)
        self.log("val/kl", kl_loss, on_step=False, on_epoch=True)
        self.log("val/beta", beta, on_step=False, on_epoch=True)

    def configure_optimizers(
        self,
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=self.lr_optimizer_factor,
            patience=self.lr_optimizer_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class AbVAEDataModule(pl.LightningDataModule):
    """
    ## Data Module that Wraps FAbVAE Data Loaders
    """

    train_ds: BaseDataLoader
    val_ds: BaseDataLoader

    def __init__(
        self,
        train_data_loader: BaseDataLoader,
        validation_data_loader: BaseDataLoader,
        batch_size: int,
        n_workers: int = 4,
    ):
        super().__init__()
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.batch_size = batch_size
        self.n_workers = n_workers

    def setup(self, stage: Optional[str] = None):
        """
        ## Load Data
        """
        self.train_ds = self.train_data_loader
        self.val_ds = self.validation_data_loader

    def train_dataloader(self) -> Any:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self) -> Any:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=True,
            persistent_workers=True
        )


class EpochSummaryPrinter(pl.Callback):
    """
    ## Prints Summary of Metrics from Epoch
    """

    def _get(self, metrics, key):
        return metrics.get(key) or metrics.get(f"{key}_epoch") or metrics.get(f"{key}/epoch")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics

        def func(x):
            return float(x) if x is not None else float("nan")

        tl = func(self._get(metrics, "train/loss"))
        tr = func(self._get(metrics, "train/recon"))
        tk = func(self._get(metrics, "train/kl"))
        tb = func(self._get(metrics, "train/beta"))
        vl = func(self._get(metrics, "val/loss"))
        vr = func(self._get(metrics, "val/recon"))
        vk = func(self._get(metrics, "val/kl"))
        vb = func(self._get(metrics, "val/beta"))
        print(
            "\n"
            "-------------------------------------"
            "\n"
            f"Epoch {trainer.current_epoch:03d} | "
            f"train: loss={tl:.3f} reconstruction={tr:.3f} kl={tk:.3f} beta={tb:.3f} | "
            f"val: loss={vl:.3f} reconstruction={vr:.3f} kl={vk:.3f} beta={vb:.3f}"
            "\n"
            "-------------------------------------"
            "\n"
        )


### --------------------------------------------------------###
#                            Main                             #
### --------------------------------------------------------###


def training_parser() -> argparse.ArgumentParser:
    """
    ## CLI Parser for Training File
    Used when training file is run independently from __main__
    ### Argparse arguments:
        \t --epochs (-e): Number of epochs \n
        \t --batch_size (-b): Batch size for training\n
        \t --gpus (-n): Number of GPUs to use \n
        \t --lr (-r): Initial learning rate \n
        \t --ckpt_dir (-c): Where checkpoints are saves \n
        \t --log_dir (-l): Where to log data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size")
    parser.add_argument("-t", "--train_data", type=str, help="Path to training data")
    parser.add_argument("-v", "--validation_data", type=str, help="Path to validation data")
    parser.add_argument(
        "-n",
        "--gpus",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Number of GPUs to use",
    )
    parser.add_argument("-r", "--lr", type=float, default=3e-4, help="Starting learning rate")
    parser.add_argument(
        "-c",
        "--ckpt_dir",
        type=Optional[str],
        default=None,
        help="Path to where checkpoints are saved",
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        type=str,
        default="runs/vae",
        help="Path to where logs are saved",
    )
    return parser


def main() -> None:
    """
    ## Main program for training
    Contains argparse and starts the workers.
    ### Argparse arguments:
        \t --epochs (-e): Number of epochs \n
        \t --batch_size (-b): Batch size for training\n
        \t --gpus (-n): Number of GPUs to use \n
        \t --lr (-r): Initial learning rate \n
        \t --ckpt_dir (-c): Where checkpoints are saves \n
        \t --log_dir (-l): Where to log data
    """

    # Device
    device = (
        torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    )

    # Argparse
    parser = training_parser()
    args = cast(ArgsTypes, parser.parse_args())

    # Setup Lightning
    core_model = AbVAEBase().to(device)
    print(core_model)
    lightning_module = VAEModule(core_model, learning_rate=args.lr)

    # KL Scheduler
    kl_scheduler = KLTargetScheduler(
        model=core_model,
        start_kl=0.1,
        min_kl=0.3,
        max_kl=1.5,
        warmup_epochs=5,
        patience_plateau=4,
        patience_ramp=3,
    )
    kl_callback = KLTargetCallback(kl_scheduler)

    # Data
    training_data = ProteinSequenceLoader(
        directory=args.train_data,
        subsample=1000,
        tokeniser=one_hot_tokenizer(sequence_length=140, device=device),
    )
    validation_data = ProteinSequenceLoader(
        directory=args.validation_data,
        subsample=200,
        tokeniser=one_hot_tokenizer(sequence_length=140, device=device),
    )
    data = AbVAEDataModule(
        train_data_loader=training_data,
        validation_data_loader=validation_data,
        batch_size=args.batch_size,
        n_workers=args.gpus,
    )

    # Logging
    logger = TensorBoardLogger(save_dir=args.log_dir, name="FAbVAE")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="{epoch:03d}--{val_loss:.3f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() and args.gpus > 0 else "cpu",
        devices=args.gpus if torch.cuda.is_available() and args.gpus > 0 else 1,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            learning_rate_monitor,
            kl_callback,
            EpochSummaryPrinter(),
        ],
        log_every_n_steps=10,
    )
    trainer.fit(model=lightning_module, datamodule=data)


### --------------------------------------------------------###
#                         Schedulers                          #
### --------------------------------------------------------###


class Scheduler(ABC):
    """
    ## Abstract Scheduler Class
    """

    def __init__(
        self,
        model: AbVAEBase,
    ):
        self.model = model

    @abstractmethod
    def step(self, epoch: int, loss: float):
        """
        ## Abstract method step
        Defines what to do at each scheduler step
        """

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]):
        """
        ## Load the scheduler's state.
        Ported from Pytorch lr_scheduler.py
        """


class KLTargetScheduler(Scheduler):
    """
    ## Sets Desired-KL During Training
    Acts in four phases: \n
        \t1. Warm-up: keep: "start_kl" for "warmup_epochs" \n
        \t2. Plateau hold: set to "min_kl" until recon improves no more than
        patience epochs \n
        \t3. Linear ramp: over "ramp_epochs" interpolate to "max_kl" \n
        \t4. Max-KL hold: keep "max_kl" afterwards. \n
    """

    def __init__(
        self,
        model: AbVAEBase,
        *,
        start_kl: float,
        min_kl: float,
        max_kl: float,
        warmup_epochs: int = 5,
        patience_plateau: int = 4,
        patience_ramp: int = 3,
        ramp_step: Optional[float] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(model=model)
        self.start_kl = start_kl
        self.min_kl = min_kl
        self.max_kl = max_kl
        self.warmup_epochs = warmup_epochs
        self.patience_plateau = patience_plateau
        self.patience_ramp = patience_ramp
        self.eps = eps

        self.best_reconstruction_loss: float = float("inf")
        self.no_improvement: int = 0
        self.current_phase: Literal["warmup", "plateau", "ramp", "max"] = "warmup"

        if ramp_step is not None:
            self.ramp_increment = ramp_step
        else:
            self.ramp_increment = (max_kl - min_kl) / 5.0

        # initialise
        self.model.kl_target = start_kl

    # -----------------------------------------------------
    def step(self, epoch: int, loss: float) -> None:
        """
        ## Scheduler Step Called Once Per Epoch
        Should use --> Validation <-- reconstruction loss!

        ### Arguments:
            \t epoch {int} -- Current epoch \n
            \t loss {float} -- Current validation
            reconstruction loss \n
        """

        # Warmup Block
        if self.current_phase == "warmup":
            # Ramp during warmup
            if epoch < self.warmup_epochs:
                progress = epoch / max(1, self.warmup_epochs - 1)
                self.model.kl_target = self.start_kl + progress * (self.min_kl - self.start_kl)
                return
            # Exit warmup phase
            elif epoch >= self.warmup_epochs:
                self.current_phase = "plateau"
                self.model.kl_target = self.min_kl
                self.best_reconstruction_loss = loss
                self.no_improvement = 0
            return

        # Plateau block
        if self.current_phase == "plateau":
            self.check_improvement(reconstruction_loss=loss)

            # Change phase if there is no improvement
            if self.no_improvement >= self.patience_plateau:
                self.current_phase = "ramp"
                self.no_improvement = 0
            return

        # Ramp block
        if self.current_phase == "ramp":
            # Adjust desired KL capped at max_kl
            new_target_kl = min(self.max_kl, self.model.kl_target + self.ramp_increment)
            self.model.kl_target = new_target_kl

            self.check_improvement(reconstruction_loss=loss)

            # Stop ramp conditions
            if self.no_improvement >= self.patience_ramp:
                self.current_phase = "max"

        # If current_phase is "max" then do nothing
        return

    def check_improvement(self, reconstruction_loss: float) -> None:
        """
        ## Checks if reconstruction loss is improved
        Adjusts self.best_reconstruction_loss if so
        """
        # Check if there is a better loss
        if reconstruction_loss < self.best_reconstruction_loss - self.eps:
            self.best_reconstruction_loss = reconstruction_loss
            self.no_improvement = 0
        # Count epochs of no improvement
        else:
            self.no_improvement += 1

    def state_dict(
        self,
    ) -> dict[str, Any]:
        """
        ## State dict used for writing checkpoints
        """
        return {
            "best_recon": self.best_reconstruction_loss,
            "no_improve": self.no_improvement,
            "phase": self.current_phase,
            "kl_target": self.model.kl_target,
        }

    def load_state_dict(self, state: dict[str, Any]):
        """
        ## Loads a given state for continued training
        """
        self.best_reconstruction_loss = state["best_recon"]
        self.no_improvement = state["no_improve"]
        self.current_phase = state["phase"]
        self.model.kl_target = state["kl_target"]


class KLTargetCallback(pl.Callback):
    """
    ## Integrates KLScheduler with Lightning
    """

    def __init__(self, scheduler: KLTargetScheduler):
        super().__init__()
        self.scheduler = scheduler

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        # Get metrics
        metrics = trainer.callback_metrics
        # Log val reconstruction loss in validation_step
        if "val/recon" in metrics:
            val_recon = float(metrics["val/recon"])
            self.scheduler.step(epoch, val_recon)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        checkpoint["kl_sched_state"] = self.scheduler.state_dict()

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        state = checkpoint.get("kl_sched_state")
        if state is not None:
            self.scheduler.load_state_dict(state)


### --------------------------------------------------------###
#                           Types                             #
### --------------------------------------------------------###


class ArgsTypes(argparse.Namespace):
    """
    ## Argparse type definitions
    """

    epochs: int
    batch_size: int
    gpus: int
    lr: float
    ckpt_dir: str
    log_dir: str
    train_data: str
    validation_data: str


if __name__ == "__main__":
    main()
