"""
--------------------    FAbVAE     ----------------------
FAbVAE is a ...
Author: Lucas Schaus
--------------------  Model Blocks ----------------------
This module contains the blocks used in constructing the
models of FAbVAE. In this file you can find:
    - ByteNet Block
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as Functionals


class ByteNetBlock(nn.Module):
    """
    ## ByteNet Block
    ByteNet-style residual block for 1D sequences.
    --> ADD CITATION FROM KEVIN YANG AND DEEPMIND PAPER

    ### Arguments
        \t- channels {int} -- Number of input/output channels.
        \t- dilation {int} -- Dilation factor for the 1D convolution.
        \t- kernel_size {int} -- Size of the convolution kernel (default 3).
        \t- dropout {float} -- Dropout probability after activation.
        \t- gated {bool} -- Whether to use a gated activation (tanh x sigmoid).
    """

    def __init__(
        self,
        *,
        channels: int,
        dilation: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.1,
        gated: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.gated = gated

        conv_out = 2 * channels if gated else channels
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=conv_out,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        ## Forward Pass of ByteNet Block
        """
        # Input shape: (B = batch_size, L = sequence_length, C = channels_in)
        residual = input_tensor
        # Project tensor to (B, C, L) for Conv1d
        input_tensor = input_tensor.transpose(1, 2)
        input_tensor = self.conv(input_tensor)

        # By default we want GeLU but keep the option open for RNN-like gating
        if self.gated:
            v, g = input_tensor.chunk(2, dim=1)
            input_tensor = torch.tanh(v) * torch.sigmoid(g)
        else:
            input_tensor = Functionals.gelu(  # pylint: disable=not-callable
                input_tensor
            )

        input_tensor = self.dropout(input_tensor)
        # Project tensor to (B, L, C) for residual addition
        input_tensor = input_tensor.transpose(1, 2)
        return self.norm(input_tensor + residual)

    def receptive_field(self) -> int:
        """
        ## Calculates the Receptive Field of ByteNet Config
        Helper function to determing the receptive field for
        a given configuration of ByteNet.
        """
        return 1 + 2 * (self.kernel_size - 1) * self.dilation


class PositionalEmbedding(nn.Module):
    """
    ## Positional Embedding
    Uses sinusoidal positional encoding scheme.

    ### Arguments:
        \t- sequence_length {int} -- Maximum sequence length at run-time
        \t- dimensions {int} -- Feature dimensions of the tensor the embedding
                                will be added to.
    """

    def __init__(self, sequence_length: int, dimensions: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Pre-compute the sin/cos values
        position = torch.arange(sequence_length).unsqueeze(1)
        divisor_term = torch.exp(
            torch.arange(0, dimensions, 2, dtype=torch.float32)
            * -(math.log(10_000.0) / dimensions)
        )

        positional_embedding = torch.zeros(sequence_length, dimensions)
        # Even indices (sin)
        positional_embedding[:, 0::2] = torch.sin(position * divisor_term)
        # Odd indices (cos)
        positional_embedding[:, 1::2] = torch.cos(position * divisor_term)

        # Register tensor
        self.positional_embedding: torch.Tensor
        self.register_buffer("positional_embedding", positional_embedding)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        ## Forward pass to include positional embeddings
        """
        # Unsqueeze to broadcast over batch
        return input_tensor + self.positional_embedding[
            : input_tensor.size(1), :
        ].unsqueeze(0)
