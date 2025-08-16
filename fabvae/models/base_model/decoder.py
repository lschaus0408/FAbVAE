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

from fabvae.modules.blocks import ByteNetBlock


class AbVAEDecoder(nn.Module):
    """Decoder for AbVAE"""

    def __init__(
        self,
        *,
        sequence_length: int = 150,
        out_channels: int = 21,
        latent_dim: int = 32,
        base_channels: int = 64,
        channel_growth_factor: int = 2,
        num_bytenet_layers: int = 1,
        activation: str = "gelu",
        bytenet_gated: bool = False,
        bytenet_dropout: float = 0.1,
        bytenet_dilation_base: int = 1,
        return_logits: bool = False,
    ) -> None:
        super().__init__()
        assert activation in {
            "gelu",
            "relu",
        }, "activation must be 'gelu' or 'relu'"
        self.seq_len = sequence_length
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.return_logits = return_logits

        activation_function = nn.GELU() if activation == "gelu" else nn.ReLU()

        # Calculate number of features
        features = 1
        # Greatest power of 2
        while features < latent_dim:
            features *= 2
        # Needs to divide sequence_length
        while sequence_length % features != 0:
            features //= 2
        self.features = features

        # Get number of upsample steps needed
        n_upsample = math.ceil(math.log2(sequence_length / features))
        self.n_upsample = n_upsample

        channels_initial = base_channels * (channel_growth_factor**n_upsample)
        self.channels_in = channels_initial

        self.dense_layers = nn.Sequential(
            nn.Linear(latent_dim, channels_initial * features),
            activation_function,
        )

        # Upsampling with stride-2
        upblocks = []
        channels_in = channels_initial
        step_bytenet_dilation = bytenet_dilation_base - 1
        for _ in range(n_upsample):
            channels_out = channels_in // channel_growth_factor
            # Upsample length ×2 and halve channels
            upblocks.append(
                nn.ConvTranspose1d(
                    channels_in,
                    channels_out,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            upblocks.append(activation_function)
            # optional ByteNet refinement blocks at current resolution
            if num_bytenet_layers:
                step_bytenet_dilation += 1
                step_bytenet_dilation = min(step_bytenet_dilation, 5)
                for _ in range(num_bytenet_layers):
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

    # ------------------------------------------------------------------
    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        ## Forward Pass of Decoder
        """
        batch_size = latent_vector.size(0)
        # Increase latent dimension to prepare for reshape (B, channels_initial * features)
        up_tensor = self.dense_layers(latent_vector)
        # Reshape to (B, channels_initial, features)
        up_tensor = up_tensor.view(batch_size, self.channels_in, self.features)
        # Upsample to (B, channels_final, sequence_length)
        up_tensor = self.up_layers(up_tensor)
        # Project to (B, 21, sequence_length)
        logits: torch.Tensor = self.sequence_projection(up_tensor)  # (B, 21, L)
        # Project to output shape (B, sequence_length, 21)
        logits = logits.transpose(1, 2)
        # Return output
        if self.return_logits:
            return logits
        return Functionals.softmax(logits, dim=-1)


class TransposeTensor(torch.nn.Module):
    """
    ## Transposes a Tensor
    Used to do the [B, L, C] <-> [B, C, L] transpositions.
    It was necessary to put this in a module to use in nn.Sequential
    """

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        ## Transpose defined as a torch module
        """
        return input_tensor.transpose(1, 2)
