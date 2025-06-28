"""
--------------------    FAbVAE     ----------------------
FAbVAE is a ...
Author: Lucas Schaus
--------------------  Model Blocks ----------------------
This module contains the blocks used in constructing the
models of FAbVAE. In this file you can find:
    - ByteNet Block
"""

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
