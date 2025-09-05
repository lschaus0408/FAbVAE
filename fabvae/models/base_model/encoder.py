"""
--------------------    FAbVAE     ----------------------
FAbVAE is a ...
Author: Lucas Schaus
-------------------- AbVAE Encoder ----------------------
Encoder for AbVAE.
Model Overview:
(Batch, Length=150, dims=21) one-hot encoded sequence
  - 1 x 1 Conv                    :   21  → N  channels
  - 7 x ByteNetBlock             : dilations 1-2-4-8-16-32-64  (bidirectional)
  - Down-sampling stack (stride-2): Nx2 → Nx4 → Nx8 → Nx16 … until L' ≤ latent_dim
  - Flatten                       :  (B, Nx16 x L')  # for L'=19 in default cfg
  - Dense layers                  :  function_mu, function_logvar  → latent_dim (default 32)
  - Reparameterisation            :  z = mu + logvar·N(0, I)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from fabvae.modules.blocks import ByteNetBlock, PositionalEmbedding


class AbVAEEncoder(nn.Module):
    """Encoder for AbVAE"""

    def __init__(
        self,
        *,
        sequence_length: int = 150,
        in_channels: int = 21,
        base_channels: int = 256,
        latent_dim: int = 32,
        num_bytenet_layers: int = 2,
        bytenet_dropout: float = 0.1,
        channel_growth_factor: int = 2,
        use_bottleneck: bool = False,
        bytenet_gated: bool = False,
        positional_embedding: bool = False,
        static_dilation: bool = True,
    ) -> None:
        super().__init__()

        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.channel_growth_factor = channel_growth_factor
        self.use_bottleneck = use_bottleneck

        # 1d conv to move from one‑hot into feature space
        self.input_projection = nn.Conv1d(in_channels, base_channels, kernel_size=1).to("cuda")

        # Positional Embedding
        if positional_embedding:
            self.pos_embed = PositionalEmbedding(
                sequence_length=sequence_length, dimensions=in_channels
            )
        else:
            self.pos_embed = nn.Identity()

        # ByteNet Layers
        if static_dilation:
            dilations = [5 for _ in range(num_bytenet_layers)]
        else:
            dilations = [2**i for i in range(num_bytenet_layers)]
        self.bytenet = nn.ModuleList(
            [
                ByteNetBlock(
                    channels=base_channels,
                    dilation=dilation,
                    dropout=bytenet_dropout,
                    gated=bytenet_gated,
                )
                for dilation in dilations
            ]
        )

        # Conv1d Down-sampling (Channel up-sampling)
        down_layers, current_latent_dimension, channel_in = (
            [],
            sequence_length,
            base_channels,
        )
        while current_latent_dimension > latent_dim:
            channel_out = channel_in * channel_growth_factor
            # CONSIDER SINGLE LAYER BOTTLENECK INSTEAD OF A ONE IN THE CHANNELS AT EACH DOWN-STEP
            if use_bottleneck:
                down_layers.append(
                    nn.Sequential(
                        nn.Conv1d(channel_in, channel_in // 2, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv1d(
                            channel_in // 2,
                            channel_in // 2,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                        nn.ReLU(),
                        nn.Conv1d(channel_in // 2, channel_out, kernel_size=1),
                        nn.ReLU(),
                    )
                )
            else:
                down_layers.append(
                    nn.Sequential(
                        nn.Conv1d(
                            channel_in,
                            channel_out,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                        nn.ReLU(),
                    )
                )
            current_latent_dimension = math.ceil(current_latent_dimension / 2)
            channel_in = channel_out
        self.down_block = nn.Sequential(*down_layers)
        self.final_len = current_latent_dimension - 1
        self.final_channels = channel_in
        self.n_down_layers = len(down_layers)

        # Latent vector parameters
        flattened_dimension = self.final_channels * self.final_len
        self.layers_mu = nn.Linear(flattened_dimension, latent_dim)
        self.layers_logvar = nn.Linear(flattened_dimension, latent_dim)

    @staticmethod
    def _reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        ## Reparametrization Trick
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, input_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ## Forward Pass of Encoder
        """
        _, sequence_length, __ = (
            input_tensor.shape
        )  # (B = batch_size, L = sequence_length, C = channels_in)
        assert sequence_length == self.sequence_length, (
            f"expected sequence length {self.sequence_length}, got {sequence_length}"
        )

        # Set positional embedding
        input_tensor = self.pos_embed(input_tensor)

        # Project tensor to (B, C, L) for Input Conv1d
        input_tensor = input_tensor.transpose(1, 2)
        input_tensor = self.input_projection(input_tensor)
        # Project tensor to (B, L, C_out) for ByteNet
        input_tensor = input_tensor.transpose(1, 2)

        # Setup ByteNet Layers
        for block in self.bytenet:
            input_tensor = block(input_tensor)

        # Project tensor to (B, C_out, L) for Down-Sampling
        input_tensor = input_tensor.transpose(1, 2)

        # Setup Down-Sampling Blocks
        if self.down_block:
            input_tensor = self.down_block(input_tensor)

        # Flatten tensor to sample latent vectors to (B, flatten_dimension)
        input_tensor = input_tensor.flatten(start_dim=1)
        mu = self.layers_mu(input_tensor)
        logvar = self.layers_logvar(input_tensor)
        z = self._reparameterise(mu, logvar)
        return z, mu, logvar
