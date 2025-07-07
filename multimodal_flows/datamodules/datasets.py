import torch
from dataclasses import dataclass
from collections import namedtuple
from torch.utils.data import Dataset

from tensorclass import TensorMultiModal

@dataclass
class DataCoupling:
    source: TensorMultiModal = None
    target: TensorMultiModal = None
    context: TensorMultiModal = None

    def __len__(self):
        return len(self.target)

    @property
    def ndim(self):
        return self.target.ndim

    @property
    def shape(self):
        return self.target.shape

    @property
    def has_source(self):
        if self.source:
            return True
        return False

    @property
    def has_target(self):
        if self.target:
            return True
        return False

    @property
    def has_context(self):
        if self.context:
            return True
        return False

class MultiModalDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.attribute = []

        # ...source (optional)

        if hasattr(self.data, "source"):

            if self.data.source.has_continuous:
                self.attribute.append("source_continuous")
                self.source_continuous = self.data.source.continuous

            if self.data.source.has_discrete:
                self.attribute.append("source_discrete")
                self.source_discrete = self.data.source.discrete

            if self.data.source.has_discrete or self.data.source.has_continuous:
                self.attribute.append("source_mask")
                self.source_mask = self.data.source.mask
                self.len = len(self.data.source)

        # ...target

        if hasattr(self.data, "target"):

            if self.data.target.has_continuous:
                self.attribute.append("target_continuous")
                self.target_continuous = self.data.target.continuous

            if self.data.target.has_discrete:
                self.attribute.append("target_discrete")
                self.target_discrete = self.data.target.discrete

            if self.data.target.has_discrete or self.data.target.has_continuous:
                self.attribute.append("target_mask")
                self.target_mask = self.data.target.mask
                self.len = len(self.data.target)

        # ...context (optional) TODO!!!

        if hasattr(self.data, "context_continuous"):
            self.attribute.append("context_continuous")
            self.context_continuous = self.data.context_continuous

        if hasattr(self.data, "context_discrete"):
            self.attribute.append("context_discrete")
            self.context_discrete = self.data.context_discrete

        self.databatch = namedtuple("databatch", self.attribute)

    def __getitem__(self, idx):
        return self.databatch(*[getattr(self, attr)[idx] for attr in self.attribute])

    def __len__(self):
        return self.len

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


def data_coupling_collate_fn(batch: namedtuple) -> DataCoupling:
    """Custom collate function for data coupling with hybrid states."""
    source = TensorMultiModal()
    target = TensorMultiModal()
    context = TensorMultiModal()

    source.continuous = (
        torch.stack([data.source_continuous for data in batch])
        if hasattr(batch[0], "source_continuous")
        else None
    )
    source.discrete = (
        torch.stack([data.source_discrete for data in batch])
        if hasattr(batch[0], "source_discrete")
        else None
    )
    source.mask = (
        torch.stack([data.source_mask for data in batch])
        if hasattr(batch[0], "source_mask")
        else None
    )
    target.continuous = (
        torch.stack([data.target_continuous for data in batch])
        if hasattr(batch[0], "target_continuous")
        else None
    )
    target.discrete = (
        torch.stack([data.target_discrete for data in batch])
        if hasattr(batch[0], "target_discrete")
        else None
    )
    target.mask = (
        torch.stack([data.target_mask for data in batch])
        if hasattr(batch[0], "target_mask")
        else None
    )
    context.continuous = (
        torch.stack([data.context_continuous for data in batch])
        if hasattr(batch[0], "context_continuous")
        else None
    )
    context.discrete = (
        torch.stack([data.context_discrete for data in batch])
        if hasattr(batch[0], "context_discrete")
        else None
    )

    return DataCoupling(source, target, context)
