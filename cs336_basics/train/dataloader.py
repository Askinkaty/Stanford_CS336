import os
import typing
from pathlib import Path
from collections.abc import Iterator

import torch
import numpy as np
from jaxtyping import Int
from torch import Tensor
from typing import Any, Dict, List


class Dataset:
    def __init__(self,
                 path_to_data: str,
                 context_length,
                 device,
                 seed = 42
                 ) -> None:
        if isinstance(path_to_data, np.ndarray):
            self.data = path_to_data
        else:
            self.data = np.load(path_to_data, mmap_mode="r")
        self.context_length = context_length
        self.total_length = self.data.shape[0]
        self.device = device
        self.seed = seed

    def __len__(self) -> int:
        return self.total_length - self.context_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i_finish = idx + self.context_length + 1
        # Token ids should stay integer for embedding/indexing.
        chunk = self.data[idx:i_finish].astype(np.int32, copy=False)
        input = torch.from_numpy(chunk[:-1]).to(self.device)
        target = torch.from_numpy(chunk[1:]).to(self.device)
        return input, target

    def get_iterator(self, batch_size: int) -> Iterator[tuple[Int[Tensor, "B C"], Int[Tensor, "B C"]]]:
        indices = np.arange(len(self), step=batch_size * self.context_length)
        for i in indices:
            batch_inputs = torch.empty((batch_size, self.context_length), dtype=torch.int32, device=self.device)
            batch_targets = torch.empty((batch_size, self.context_length), dtype=torch.int32, device=self.device)
            for b in range(batch_size):
                inputs, targets = self.__getitem__(i.item() + b)
                batch_inputs[b] = inputs
                batch_targets[b] = targets
            yield batch_inputs, batch_targets


    def get_batch(self, batch_size: int) -> tuple[Int[Tensor, "B C"], Int[Tensor, "B C"]]:
        batch_inputs = torch.empty((batch_size, self.context_length), dtype=torch.int32, device=self.device)
        batch_targets = torch.empty((batch_size, self.context_length), dtype=torch.int32, device=self.device)
        indices = torch.randint(0, len(self), size=(batch_size,))
        for b in range(batch_size):
            inputs, targets = self.__getitem__(indices[b].item())
            batch_inputs[b] = inputs
            batch_targets[b] = targets
        return batch_inputs, batch_targets


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, out)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> int:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch
