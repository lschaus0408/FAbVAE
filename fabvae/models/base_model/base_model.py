"""
--------------------    FAbVAE     ----------------------
FAbVAE is a ...
Author: Lucas Schaus
-------------------  Base Model ABC ---------------------
Abstracts the Base Model.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class FAbVAEBase(ABC, nn.Module):
    """
    ## Abstract Class of a FAbVAE Model
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    @abstractmethod
    def elbo(
        self,
        input_embedding: torch.Tensor,
        position_weights: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ## Abstract Method For Loss Calculation
        """
