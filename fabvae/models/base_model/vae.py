"""
--------------------    FAbVAE     ----------------------
FAbVAE is a ...
Author: Lucas Schaus
------------------ AbVAE Base Model ---------------------
AbVAE Base Model. Variational Auto-Encoder that stitches the
ByteNet-based encoder and decoder together.
Model Overview:
(Batch, Length=150, dims=21) one-hot encoded sequence
  - Encoder -> z (latent_dimension)
  - z -> Decoder
  - Decoder -> (Batch, sequence_length, 21)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as Functionals

# Local modules created in the canvas
from fabvae.models.base_model.encoder import AbVAEEncoder
from fabvae.models.base_model.decoder import AbVAEDecoder


class AbVAEBase(nn.Module):
    """Base VAE of AbVAE"""

    def __init__(
        self,
        *,
        sequence_length: int = 150,
        in_channels: int = 21,
        latent_dim: int = 32,
        base_channels: int = 128,
        # PID / KL control (ADJUST BASE-PARAMS)
        kl_target: float = 1.0,
        beta_init: float = 0.0,
        beta_min: float = 0.0,
        beta_max: float = 1.0,
        kl_p: float = 0.01,
        kl_i: float = 0.001,
        kl_d: float = 0.0,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        encoder_kwargs = encoder_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}

        self.encoder = AbVAEEncoder(
            seq_len=sequence_length,
            in_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            **encoder_kwargs,
        )

        self.decoder = AbVAEDecoder(
            seq_len=sequence_length,
            out_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            return_logits=True,
            **decoder_kwargs,
        )

        # PID Algorithm Params
        self.beta: torch.Tensor
        self._internal_error_sum: torch.Tensor
        self._previous_error: torch.Tensor
        self.register_buffer("beta", torch.tensor(beta_init))
        self.register_buffer("_internal_error_sum", torch.tensor(0.0))
        self.register_buffer("_previous_error", torch.tensor(0.0))
        self.kl_target = kl_target
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.kl_p, self.kl_i, self.kl_d = kl_p, kl_i, kl_d

        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.latent_dim = latent_dim

    def forward(
        self,
        input_embedding: torch.Tensor,
        *,
        sample: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ## Forward pass of AbVAE Base

        ### Arguments
            \t- input_embedding {torch.Tensor} -- One-hot encoded input sequence.
            \t- sample {bool} -- Whether to sample z (True) or use the mean (False).
            \t- return_logits {bool} -- If True, return raw decoder logits instead of probabilities.
        """
        latent_vector, mu, logvar = self.encoder(input_embedding)

        # Sample the mean instead of z
        if not sample:
            latent_vector = mu

        logits = self.decoder(latent_vector)
        if return_logits:
            return logits, mu, logvar
        probabilities = Functionals.softmax(logits, dim=-1)
        return probabilities, mu, logvar

    def elbo(
        self,
        input_embedding: torch.Tensor,
        position_weights: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ## Compute the Evidence Lower BOund (ELBO)
        Weigths reconstruction based on position_weights. Epsilon is there
        to handle -inf logits. Calculates KL divergence loss and the PID-
        calculated beta to weight the KL divergence.
        Returns reconstruction term, KL term, and their sum (negative ELBO).
        """
        logits, mu, logvar = self.forward(input_embedding, return_logits=True)

        # Weighted reconstruction loss
        log_probabilities = Functionals.log_softmax(logits + eps, dim=-1)
        reconstruction_per_position = -torch.sum(
            input_embedding * log_probabilities, dim=-1
        )

        if position_weights is not None:
            # For batch-wise position weights
            if position_weights.dim() == 1:
                position_weights = position_weights.unsqueeze(0)
            reconstruction_per_position = (
                reconstruction_per_position
                * position_weights.to(reconstruction_per_position.device)
            )

        # Average reconstruction per batch and on each position
        reconstruction_loss = reconstruction_per_position.mean()

        # KL-Divergence Loss
        kl_divergence_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
        )
        kl_divergence_loss = kl_divergence_loss.mean()

        #  PID algorithm
        delta_error = kl_divergence_loss.detach() - self.kl_target
        self._internal_error_sum += delta_error
        derivative = delta_error - self._previous_error
        self._previous_error = delta_error
        beta_new = (
            self.beta
            + self.kl_p * delta_error
            + self.kl_i * self._internal_error_sum
            + self.kl_d * derivative
        )
        self.beta = beta_new.clamp_(self.beta_min, self.beta_max)

        loss = reconstruction_loss + self.beta * kl_divergence_loss
        return loss, reconstruction_loss, kl_divergence_loss, self.beta.detach().clone()
