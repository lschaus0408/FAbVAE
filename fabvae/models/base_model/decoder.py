"""
--------------------    FAbVAE     ----------------------
FAbVAE is a ...
Author: Lucas Schaus
-------------------- AbVAE Decoder ----------------------
Decoder for AbVAE.
Model Overview:
(Batch, latent_dim) z
  - Dense + GELU        : latent_dim  -> init_channels *  features  (flat)
  - reshape             : (Batch, init_channels, features)
  - n_upsample x {Upsample x2 + ByteNetBlock}  channels halved each step
  - 1 x 1 Conv          : channels -> 21
  - Softmax(dim=-1)     : (Batch, sequence_length, 21)

    - features is calculated as the greates power of 2 that is less than latent_dim and
      divides sequence_length
    - n_upsample is inferred to reach "sequence_length"
    - Use "return_logits=True" if you prefer raw logits instead of probabilities.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as Functionals

from fabvae.modules.blocks import (
    ByteNetBlock,
    TransposeTensor,
    UpBlock,
    NormType,
    ActivationType,
)


class AbVAEDecoder(nn.Module):
    """Decoder for AbVAE"""

    def __init__(
        self,
        *,
        sequence_length: int = 150,
        out_channels: int = 21,
        latent_dim: int = 32,
        base_channels: int = 64,
        num_bytenet_layers_per_upblock: int = 2,
        num_bytenet_layers_total: int = 3,
        activation: ActivationType = "gelu",
        bytenet_gated: bool = False,
        bytenet_dropout: float = 0.1,
        bytenet_dilation_base: int = 2,
        seed_length: int = 2,
        seed_total_units: int = 512,
        norm: NormType = "batch",
        add_second_convolution_per_up: bool = False,
        final_adjustment: bool = True,
        return_logits: bool = False,
    ) -> None:
        super().__init__()
        # Set Params
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.return_logits = return_logits

        if activation == "gelu":
            activation_function = nn.GELU()
        elif activation == "relu":
            activation_function = nn.ReLU()
        else:
            raise ValueError(f"Activation must be 'gelu' or 'relu', not {activation}")

        # Get number of upsample steps needed
        # Double the lenght per step -> remainder is handled at the end
        n_upsample = math.floor(math.log2(max(1, sequence_length // seed_length)))
        self.n_upsample = n_upsample

        # Get number of channels at each step
        seed_channels = max(1, seed_total_units // seed_length)
        channels_schedule = [seed_channels]
        for _ in range(n_upsample):
            next_channel = max(self.base_channels, channels_schedule[-1] // 2)
            channels_schedule.append(next_channel)

        # First dense step from latent dim
        dense_out_units = channels_schedule[0] * seed_length
        self.dense_layers = nn.Sequential(
            nn.Linear(latent_dim, dense_out_units),
            activation_function,
        )

        # Build Up Sampling Model
        upblocks: list[nn.Module] = []
        step_bytenet_dilation = bytenet_dilation_base - 1
        add_bytenet_iteration = n_upsample - num_bytenet_layers_total

        channels_in = channels_schedule[0]
        for iteration in range(n_upsample):
            # Out is always the next steps In
            channels_out = channels_schedule[iteration + 1]
            # Upsample length ×2 and halve channels
            if iteration == 0:
                kernel_size_refine = 5
            else:
                kernel_size_refine = 3
            upblocks.append(
                UpBlock(
                    in_channels=channels_in,
                    out_channels=channels_out,
                    kernel_size_refine=kernel_size_refine,
                    activation=activation,
                    norm=norm,
                    second_convolution=add_second_convolution_per_up,
                )
            )

            # optional ByteNet refinement blocks at current resolution
            if iteration >= add_bytenet_iteration:
                step_bytenet_dilation += 1
                step_bytenet_dilation = min(step_bytenet_dilation, 5)
                for _ in range(num_bytenet_layers_per_upblock):
                    upblocks.append(TransposeTensor())
                    upblocks.append(
                        ByteNetBlock(
                            channels=channels_out,
                            dilation=step_bytenet_dilation,
                            dropout=bytenet_dropout,
                            gated=bytenet_gated,
                        )
                    )
                    upblocks.append(TransposeTensor())
            channels_in = channels_out

        self.up_layers = nn.Sequential(*upblocks)

        # Projection to sequence shape
        self.sequence_projection = nn.Conv1d(channels_in, out_channels, kernel_size=1)

        # Final pool and convolution
        self.final_adjustment = final_adjustment
        if not self.final_adjustment:
            self.pool = nn.AdaptiveAvgPool1d(self.sequence_length)
            self.final_conv = nn.Conv1d(
                in_channels=self.base_channels,
                out_channels=self.base_channels,
                kernel_size=3,
                padding=1,
            )

        self._seed_length = seed_length
        self._seed_channels = channels_schedule[0]

    # ------------------------------------------------------------------
    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        ## Forward Pass of Decoder
        """
        batch_size = latent_vector.size(0)

        # Increase latent dimension to prepare for reshape (B, latent_dimension)
        up_tensor: torch.Tensor = self.dense_layers(latent_vector)
        # Reshape to (B, channels_initial, features)
        up_tensor = up_tensor.view(batch_size, self._seed_channels, self._seed_length)

        # Upsample to (B, channels_final, sequence_length)
        up_tensor = self.up_layers(up_tensor)

        # Final Convolution
        if self.final_adjustment:
            up_tensor = Functionals.interpolate(
                up_tensor, size=self.sequence_length, mode="nearest"
            )
        else:
            up_tensor = self.pool(up_tensor)
            up_tensor = self.final_conv(up_tensor)

        # Project to (B, 21, sequence_length)
        logits: torch.Tensor = self.sequence_projection(up_tensor)  # (B, 21, L)
        # Project to output shape (B, sequence_length, 21)
        logits = logits.transpose(1, 2)
        # Return output
        if self.return_logits:
            return logits
        return Functionals.softmax(logits, dim=-1)
