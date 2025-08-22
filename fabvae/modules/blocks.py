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

from typing import Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as Functionals

ActivationType: TypeAlias = Literal["gelu", "relu"]
NormType: TypeAlias = Literal["batch", "layer", "none"]


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
        self.conv_dilated = nn.Conv1d(
            in_channels=channels,
            out_channels=conv_out,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.conv_normal = nn.Conv1d(
            in_channels=channels,
            out_channels=conv_out,
            kernel_size=1,
            dilation=1,
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

        ### Normal Convolution Sub-Block ###
        input_tensor = self.one_by_one_convolution_block(input_tensor)
        # Project tensor to (B, L, C) for norm across channels
        input_tensor = input_tensor.transpose(1, 2)

        ### High Dilation Convolution Sub-Block ###
        input_tensor = self.high_dilation_convolution_block(input_tensor)
        # Project tensor to (B, L, C) for norm across channels
        input_tensor = input_tensor.transpose(1, 2)

        ### Normal Convolution Sub-Block ###
        input_tensor = self.one_by_one_convolution_block(input_tensor)
        # Project tensor to (B, L, C) for residual addition
        input_tensor = input_tensor.transpose(1, 2)

        input_tensor = self.dropout(input_tensor)

        return input_tensor + residual

    def one_by_one_convolution_block(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        ## Performs the 1x1 convolution block of ByteNet
        """
        input_tensor = self.norm(input_tensor)
        # By default we want GeLU but keep the option open for RNN-like gating
        if self.gated:
            v, g = input_tensor.chunk(2, dim=1)
            input_tensor = torch.tanh(v) * torch.sigmoid(g)
        else:
            input_tensor = Functionals.gelu(  # pylint: disable=not-callable
                input_tensor
            )
        # Project tensor to (B, C, L) for Conv1d
        input_tensor = input_tensor.transpose(1, 2)
        input_tensor = self.conv_normal(input_tensor)
        return input_tensor

    def high_dilation_convolution_block(
        self, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        ## Performs high-dilation block of ByteNet
        """
        input_tensor = self.norm(input_tensor)
        # By default we want GeLU but keep the option open for RNN-like gating
        if self.gated:
            v, g = input_tensor.chunk(2, dim=1)
            input_tensor = torch.tanh(v) * torch.sigmoid(g)
        else:
            input_tensor = Functionals.gelu(  # pylint: disable=not-callable
                input_tensor
            )
        # Project tensor to (B, C, L) for Conv1d
        input_tensor = input_tensor.transpose(1, 2)
        input_tensor = self.conv_dilated(input_tensor)

        # Pad to preserve shape
        pad_total = (self.kernel_size - 1) * self.dilation
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return Functionals.pad(input_tensor, (pad_left, pad_right))

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


class DepthwiseSeparableConvolution(nn.Module):
    """
    ## Depth-wise Separable 1D Convolution
    Depth-wise followed by point-wise 1D Convolution.
    --> Cite MobileNet (Howard et al.)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = False,
        norm: NormType = "batch",
        activation: ActivationType = "gelu",
    ):
        super().__init__()
        # Define Convolutions
        padding = (kernel_size // 2) * dilation
        self.depth_wise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.point_wise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

        # Define Norms
        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = None

        self._use_layer_norm = norm == "layer"

        # Define Activations
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Activation must be 'gelu' or 'relu', not {activation}")

    def forward(self, tensor_input: torch.Tensor) -> torch.Tensor:
        """
        ## Forward Pass
        """
        tensor_input = self.depth_wise(tensor_input)
        tensor_input = self.point_wise(tensor_input)
        if self.norm is not None:
            if self._use_layer_norm:
                # Layer Norm requires [B, C, L] -> [B, L, C]
                tensor_input = tensor_input.transpose(1, 2)
                tensor_input = self.norm(tensor_input)
                tensor_input = tensor_input.transpose(1, 2)
            else:
                tensor_input = self.norm(tensor_input)
        tensor_input = self.activation(tensor_input)
        return tensor_input


class UpBlock(nn.Module):
    """
    ## UpBlock Module for FAbVAE Decoders
    Nearest-neighbor upsample by 2x, followed by depth-wise-separable
    convolution to refine. Lastly, a 1x1 projection to halve channel size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_refine: int = 5,
        kernel_size_project: int = 3,
        activation: ActivationType = "gelu",
        norm: NormType = "batch",
        second_convolution: bool = False,
    ):
        super().__init__()
        # Convolutions
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.refine = DepthwiseSeparableConvolution(
            in_channels,
            in_channels,
            kernel_size=kernel_size_refine,
            dilation=1,
            norm=norm,
            activation=activation,
        )
        self.project = DepthwiseSeparableConvolution(
            in_channels,
            out_channels,
            kernel_size=kernel_size_project,
            dilation=1,
            norm=norm,
            activation=activation,
        )
        if second_convolution:
            self.second_convolution = DepthwiseSeparableConvolution(
                out_channels,
                out_channels,
                kernel_size=kernel_size_project,
                dilation=1,
                norm=norm,
                activation=activation,
            )
        else:
            self.second_convolution = None

    def forward(self, tensor_input: torch.Tensor) -> torch.Tensor:
        """
        ## Forward Pass
        """
        tensor_input = self.upsample(tensor_input)
        tensor_input = self.refine(tensor_input)
        tensor_input = self.project(tensor_input)
        if self.second_convolution is not None:
            tensor_input = self.second_convolution(tensor_input)
        return tensor_input
