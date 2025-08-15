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

import argparse
from pathlib import Path
from typing import Tuple, Optional, Literal, Any, Generator
from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing.spawn import spawn as torchmp_spawn


from fabvae.models.base_model.vae import AbVAEBase
from fabvae.data.load_sequences import MockDataLoader

### --------------------------------------------------------###
#                       Training Utils                        #
### --------------------------------------------------------###


@contextmanager
def init_distributed_gpu(rank: int, world_size: int) -> Generator[Any]:
    """
    ## Initialise default process group for multi-GPU
    Use Distributed Data Parallel for multi-GPU
    ### Arguments:
        \t rank {int} -- GPU identifier number [0, ... world_size -1]
        \t world_size {int} -- Number of total GPUs
    """
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    # Exit gracefully even if the program crashes
    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def cleanup_distributed() -> None:
    """
    ## Cleanup DDP
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(state: dict, checkpoint_directory: Path, name: str) -> None:
    """
    ## Save Checkpoint States
    """
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_directory / name)


def load_latest_checkpoint(
    model: AbVAEBase,
    optimiser: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    kl_scheduler: Scheduler,
    checkpoint_directory: Path,
) -> Tuple[int, float]:
    """
    ## Load Some Previous Checkpoint
    If no checkpoint is found, start from the beginning
    """
    checkpoint_path = checkpoint_directory / "latest.pt"
    # Load Checkpoint if it exists
    if checkpoint_path.is_file():
        checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        optimiser.load_state_dict(checkpoint["optim_state"])
        kl_scheduler.load_state_dict(checkpoint["kl_sched_state"])
        lr_scheduler.load_state_dict(checkpoint["sched_state"])
        return checkpoint["epoch"], checkpoint.get("best_val", float("inf"))

    # Start from the beginning
    return 0, float("inf")


### --------------------------------------------------------###
#                           Training                          #
### --------------------------------------------------------###


def train_one_epoch(
    model: AbVAEBase,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """
    ## One Epoch Cycle
    Loads training data to device, performs forward and backwards passes.
    Updates loss trackers.
    ### Arguments:
        model {AbVAEBase} -- The model to be trained \n
        loader {DataLoader} -- Loads batches onto the model \n
        optimizer {} -- Optimizer to be used \n
        device {torch.device} -- Device to train on \n

    ### Returns:
        \t {tuple}: \n
            \t total_loss \n
            \t reconstruction_loss \n
            \t kl_loss \n
    """
    # Setup
    model.train()
    total_loss: float = 0.0
    total_recon: float = 0.0
    total_kl: float = 0.0

    for batch in loader:
        # Load params
        batch: torch.Tensor = batch.to(device)
        optimizer.zero_grad()

        # Train and backprop
        loss, recon, kl, beta = model.elbo(batch)
        loss.backward()
        optimizer.step()

        # Loss registration
        batch_size = batch.size(0)
        total_loss += loss.item() * batch_size
        total_recon += recon.item() * batch_size
        total_kl += kl.item() * batch_size
        total_beta += beta.item() * batch_size

    # Average Losses
    n = len(loader.dataset)
    average_total = total_loss / n
    average_recon = total_recon / n
    average_kl = total_kl / n
    average_beta = total_beta / n

    return average_total, average_recon, average_kl, average_beta


def validate(
    model: AbVAEBase, loader: DataLoader, device: torch.device
) -> tuple[float, float, float, float]:
    """
    ## Validation Cycle
    Loads validation data to device, performs forward and backwards passes.
    Updates loss trackers.
    ### Arguments:
        model {AbVAEBase} -- The model to be trained \n
        loader {DataLoader} -- Loads batches onto the model \n
        device {torch.device} -- Device to train on \n

    ### Returns:
        \t {tuple}: \n
            \t total_loss \n
            \t reconstruction_loss \n
            \t kl_loss \n
    """
    # Setup
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    # No grad for validation
    with torch.no_grad():
        for batch in loader:
            # Load params
            batch = batch.to(device)

            # Get losses
            loss, recon, kl, beta = model.elbo(batch)

            # Register losses
            batch_size = batch.size(0)
            total_loss += loss.item() * batch_size
            total_recon += recon.item() * batch_size
            total_kl += kl.item() * batch_size
            total_beta += beta.item() * batch_size

    # Average losses
    n = len(loader.dataset)
    average_loss = total_loss / n
    average_recon = total_recon / n
    average_kl = total_kl / n
    average_beta = total_beta / n
    return average_loss, average_recon, average_kl, average_beta


### --------------------------------------------------------###
#                            Main                             #
### --------------------------------------------------------###


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

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-n", "--gpus", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("-r", "--lr", type=float, default=3e-4)
    parser.add_argument("-c", "--ckpt_dir", type=Optional[str], default=None)
    parser.add_argument("-l", "--log_dir", type=str, default="runs/vae")
    args = parser.parse_args()

    # Start workers
    world_size = args.gpus
    if world_size > 1:
        torchmp_spawn(worker, args=(world_size, args), nprocs=world_size)
    else:
        worker(0, world_size, args)


def worker(rank: int, world_size: int, args):
    """
    ## Single worker for training
    """
    # Initialize distributor
    with init_distributed_gpu(rank=rank, world_size=world_size):
        # Load GPU device
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Data Loading
        train_ds, val_ds = MockDataLoader(10_000), MockDataLoader(2_000)
        train_sampler = (
            DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True
            )
            if world_size > 1
            else None
        )
        val_sampler = (
            DistributedSampler(
                val_ds, num_replicas=world_size, rank=rank, shuffle=False
            )
            if world_size > 1
            else None
        )

        train_loader = DataLoader(
            train_ds,
            args.batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Load Model and cast to device
        model = AbVAEBase().to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[rank])

        # Setup optimizer and LR Scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        # KL Scheduler
        kl_sched = KLTargetScheduler(
            model,
            start_kl=0.1,
            min_kl=0.3,
            max_kl=1.0,
            warmup_epochs=5,
            patience_plateau=4,
            patience_ramp=3,
        )

        # Load checkpoint if it exists
        if args.ckpt_dir is not None:
            checkpoint_directory = Path(args.ckpt_dir)
            start_epoch, best_val = load_latest_checkpoint(
                model, optimizer, lr_sched, kl_sched, checkpoint_directory
            )
        else:
            start_epoch = 0
            best_val = float("inf")

        # If this is GPU 0, it is responsible for logging as well
        if rank == 0:
            writer = SummaryWriter(args.log_dir)
        else:
            writer = None

        # Start training
        for epoch in range(start_epoch, args.epochs):
            if train_loader.sampler and isinstance(
                train_loader.sampler, DistributedSampler
            ):
                train_loader.sampler.set_epoch(epoch)

            # One pass of training
            training_loss, training_recon, training_kl, training_beta = train_one_epoch(
                model, train_loader, optimizer, device
            )
            validation_loss, validation_recon, validation_kl, validation_beta = (
                validate(model, val_loader, device)
            )

            lr_sched.step(validation_loss)
            kl_sched.step(epoch, validation_recon)

            # Write the metrics to stdout if GPU 0
            if rank == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Ep {epoch:03d} | LR {lr:.2e} | β {validation_beta:.3f} \
                    | tr {training_loss:.3f} | val {validation_loss:.3f}"
                )
                if writer:
                    writer.add_scalars(
                        "Loss", {"train": training_loss, "val": validation_loss}, epoch
                    )
                    writer.add_scalars(
                        "Recon",
                        {"train": training_recon, "val": validation_recon},
                        epoch,
                    )
                    writer.add_scalars(
                        "KL", {"train": training_kl, "val": validation_kl}, epoch
                    )
                    writer.add_scalars(
                        "beta", {"train": training_beta, "val": validation_beta}, epoch
                    )
                    writer.add_scalar("lr", lr, epoch)

            # Update the user on Optimizer, KL and LR states
            if rank == 0 and (epoch % 5 == 0 or validation_loss < best_val):
                best_val = min(best_val, validation_loss)
                state = {
                    "epoch": epoch + 1,
                    "model_state": (
                        model.module.state_dict()
                        if isinstance(model, DDP)
                        else model.state_dict()
                    ),
                    "optim_state": optimizer.state_dict(),
                    "sched_state": lr_sched.state_dict(),
                    "kl_sched_state": kl_sched.state_dict(),
                    "best_val": best_val,
                }
                save_checkpoint(state, checkpoint_directory, "latest.pt")
                if validation_loss <= best_val:
                    save_checkpoint(state, checkpoint_directory, "best.pt")

        # Cleanup
        if writer:
            writer.close()


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
                self.model.kl_target = self.start_kl + progress * (
                    self.min_kl - self.start_kl
                )
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


if __name__ == "__main__":
    main()
