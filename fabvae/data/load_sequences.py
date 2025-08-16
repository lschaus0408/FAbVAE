"""
--------------------    FAbVAE     ----------------------
FAbVAE is a ...
Author: Lucas Schaus
--------------------- Data Loader -----------------------
Dataset loader for FAbVAE. Contains the following data
loaders:
- MockDataLoader: Loads random sequences of a given shape
                  and type
- ProteinSequenceLoader: Loads protein sequences of type
                         str and optionally tokenizes/embeds them
- ProteinEmbeddingLoader: Loads pre-calculated protein
                          embeddings
"""

import pickle
from pathlib import Path
from typing import Literal, Optional, Callable, TypeAlias

import torch

import pandas as pd
import numpy as np

from torch.utils.data import Dataset

TokenizerType: TypeAlias = Optional[Callable[[str], torch.Tensor]]


class BaseDataLoader(Dataset):
    """
    ## Abstraction for DataLoaders
    """

    def __init__(self, tokenizer: TokenizerType = None):
        super().__init__()
        self.tokenizer = tokenizer

    def _get_raw(self, idx):
        raise NotImplementedError(
            "_get_raw not implemented for Parent Class. "
            "If this is a child class, you need an implementation of _get_raw in the child"
        )

    def __getitem__(self, idx):
        raw = self._get_raw(idx)
        if self.tokenizer is None:
            return raw
        return self.tokenizer(raw)


class MockDataLoader(BaseDataLoader):
    """
    ## Dummy dataset returning tensors
    Shape and type of the tensor can be specified
    """

    def __init__(
        self,
        n_sequences: int = 1024,
        sequence_length: int = 150,
        features_length: int = 21,
        dtype: Literal["float", "int", "bool"] = "bool",
    ):
        super().__init__(tokenizer=None)
        self.size = n_sequences
        self.sequence_length = sequence_length
        self.features_length = features_length
        self.dtype = dtype

    def __len__(self):
        return self.size

    def _get_raw(self, idx):
        """
        Unused
        """
        return

    def __getitem__(self, idx):
        if self.dtype == "bool":
            x = torch.randint(0, self.features_length, (self.sequence_length,))
            return torch.nn.functional.one_hot(  # pylint: disable=not-callable
                x, self.features_length
            ).float()
        if self.dtype == "int":
            return torch.randint(
                0, 100, (self.sequence_length, self.features_length), dtype=torch.int32
            )
        if self.dtype == "float":
            return torch.rand(
                self.sequence_length, self.features_length, dtype=torch.float32
            )
        raise ValueError(f"dtype {self.dtype} not recognized.")


class ProteinSequenceLoader(BaseDataLoader):
    """
    ## Load Datasets from CSV Files
    """

    def __init__(
        self,
        directory: Path,
        *,
        pattern: str = "*.csv",
        sequence_column: str = "Sequence_aa",
        tokeniser: Optional[Callable[[str], torch.Tensor]] = None,
    ) -> None:
        super().__init__(tokeniser)

        # Get list of files
        self.files: list[Path] = sorted(directory.glob(pattern))

        if not self.files:
            raise FileNotFoundError(f"No files matching {pattern} in {directory}")

        # Load sequences into memory
        self.sequences: list[str] = []
        for file in self.files:
            df = pd.read_csv(file)
            if sequence_column not in df:
                raise KeyError(f"{sequence_column} column missing in {file}")
            self.sequences.extend(df[sequence_column].astype(str).tolist())

    def __len__(self):
        return len(self.sequences)

    def _get_raw(self, idx):
        return self.sequences[idx]


class ProteinEmbeddingLoader(Dataset):
    """
    ## Loads Pre-computed Tensors
    """

    def __init__(
        self,
        directory: Path,
        pattern: Literal["*.pt", "*.npy", "*.npz", "*.pkl"] = "*.pt",
    ) -> None:

        # Load files
        self.paths = sorted(directory.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"No embedding files found in {directory}")

        self.embeddings: list[torch.Tensor] = []
        for path in self.paths:
            tensors = self._load_and_split(path)
            self.embeddings.append(tensors.view(-1, *tensors.shape[-2:]))
        self.tensors = torch.cat(self.embeddings, dim=0)

    def _load_and_split(self, path: Path) -> torch.Tensor:
        """
        ## Returns a list of tensors from file
        """
        data: torch.Tensor
        if path.suffix == ".pt":
            data = torch.load(path, map_location="cpu")
        elif path.suffix == ".npy":
            data = np.load(path)
        elif path.suffix in {".pkl", ".pickle"}:
            with open(path, "rb") as file_handle:
                data = pickle.load(file_handle)
        else:
            raise ValueError(f"Unsupported format {path.suffix}")

        return data

    def __len__(self):
        return self.tensors.size(0)

    def __getitem__(self, idx):
        self.tensors[idx]  # pylint: disable=pointless-statement
