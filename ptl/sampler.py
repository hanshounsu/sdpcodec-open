import torch
import math
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from typing import List, Optional

class ResumableRandomSampler(Sampler[int]):
    """Single-process sampler with shuffle and mid-epoch resume."""
    def __init__(self, data_source: Dataset, shuffle: bool = True, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = int(seed)
        self._epoch = 0
        self._cursor = 0
        self._indices_epoch = None  # type: Optional[int]
        self._indices = []  # type: List[int]
        self._build_indices()

    def _build_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            self._indices = torch.randperm(len(self.data_source), generator=g).tolist()
        else:
            self._indices = list(range(len(self.data_source)))
        self._indices_epoch = self._epoch

    def __iter__(self):
        if self._indices_epoch != self._epoch or not self._indices:
            self._build_indices()
        for i in range(self._cursor, len(self._indices)):
            self._cursor = i + 1
            yield self._indices[i]

    def __len__(self):
        if self._indices_epoch != self._epoch or not self._indices:
            self._build_indices()
        return max(0, len(self._indices) - int(self._cursor))

    def state_dict(self):
        return {"epoch": self._epoch, "cursor": self._cursor}

    def load_state_dict(self, state):
        self._epoch = int(state.get("epoch", 0))
        self._cursor = int(state.get("cursor", 0))
        self._build_indices()

    def set_epoch(self, epoch: int):
        epoch = int(epoch)
        if epoch != self._epoch:
            self._epoch = epoch
            self._cursor = 0
            self._build_indices()
        else:
            # Same epoch called again (e.g., by Lightning on resume): keep cursor
            if self._indices_epoch != self._epoch:
                self._build_indices()


class ResumableDistributedSampler(DistributedSampler):
    """DDP sampler with shuffle and mid-epoch resume (per-rank cursor)."""
    def __init__(self, dataset: Dataset, num_replicas: int, rank: int,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, seed=seed, drop_last=drop_last)
        self._cursor = 0
        self._indices_epoch = None  # to know if current indices match self.epoch
        self._rank_indices = []  # list[int]
        self._build_indices()

    def _build_indices(self):
        # Build global indices for this epoch
        n = len(self.dataset)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        if self.drop_last:
            total_size = (n // self.num_replicas) * self.num_replicas
            indices = indices[:total_size]
        else:
            total_size = int(math.ceil(n / self.num_replicas)) * self.num_replicas
            if len(indices) < total_size:
                # pad
                indices += indices[:(total_size - len(indices))]

        # Per-rank striding
        self._rank_indices = indices[self.rank:total_size:self.num_replicas]
        self._indices_epoch = self.epoch

    def __iter__(self):
        if self._indices_epoch != self.epoch or not self._rank_indices:
            self._build_indices()
        for i in range(self._cursor, len(self._rank_indices)):
            self._cursor = i + 1
            yield self._rank_indices[i]

    def __len__(self):
        if self._indices_epoch != self.epoch or not self._rank_indices:
            self._build_indices()
        return max(0, len(self._rank_indices) - int(self._cursor))

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self._cursor}

    def load_state_dict(self, state):
        self.epoch = int(state.get("epoch", 0))
        self._cursor = int(state.get("cursor", 0))
        self._build_indices()

    def set_epoch(self, epoch: int):
        epoch = int(epoch)
        if epoch != self.epoch:
            self.epoch = epoch
            self._cursor = 0
            self._build_indices()
        else:
            # Same epoch called again (e.g., on resume): keep cursor
            if self._indices_epoch != self.epoch:
                self._build_indices()
