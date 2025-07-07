from dataclasses import dataclass
from typing import List

import h5py
import torch

""" Data classes for multi-modal states and data couplings. 
"""


@dataclass
class TensorMultiModal:
    time: torch.Tensor = None
    continuous: torch.Tensor = None
    discrete: torch.Tensor = None
    mask: torch.Tensor = None

    def __len__(self):
        if self.ndim > 0:
            return len(getattr(self, self.available_modes()[-1]))
        return 0

    def __getitem__(self, index) -> "TensorMultiModal":
        return TensorMultiModal(
            time=self.time[index] if self.time is not None else None,
            continuous=self.continuous[index] if self.continuous is not None else None,
            discrete=self.discrete[index] if self.discrete is not None else None,
            mask=self.mask[index] if self.mask is not None else None,
        )

    # Methods to manipulate the state

    def to(self, device: str):
        return self._apply_fn(lambda tensor: tensor.to(device))

    def detach(self):
        return self._apply_fn(lambda tensor: tensor.detach())

    def cpu(self):
        return self._apply_fn(lambda tensor: tensor.cpu())

    def clone(self):
        return self._apply_fn(lambda tensor: tensor.clone())

    @property
    def ndim(self):
        if not len(self.available_modes()):
            return 0
        return len(getattr(self, self.available_modes()[-1]).shape)

    @property
    def shape(self):
        if self.ndim > 0:
            return getattr(self, self.available_modes()[-1]).shape[:-1]
        return None

    def unsqueeze(self, dim: int, mode: str = None) -> "TensorMultiModal":
        """Add a singleton dimension at the specified dim for a specific mode or all modes."""
        return self._apply_tensor_op(torch.unsqueeze, dim, mode=mode)

    def squeeze(self, dim: int = None, mode: str = None) -> "TensorMultiModal":
        """Remove singleton dimensions for a specific mode or all modes."""
        return self._apply_tensor_op(torch.squeeze, dim, mode=mode)

    def reshape(self, *shape, mode: str = None) -> "TensorMultiModal":
        """Reshape a specific mode or all modes."""
        return self._apply_tensor_op(torch.reshape, *shape, mode=mode)

    def expand(self, *sizes, mode: str = None) -> "TensorMultiModal":
        """Expand a specific mode or all modes."""
        return self._apply_tensor_op(torch.Tensor.expand, *sizes, mode=mode)

    def repeat(self, *repeats, mode: str = None) -> "TensorMultiModal":
        """Repeat a specific mode or all modes."""
        return self._apply_tensor_op(torch.Tensor.repeat, *repeats, mode=mode)

    @property
    def has_discrete(self):
        if "discrete" in self.available_modes():
            return True
        return False

    @property
    def has_continuous(self):
        if "continuous" in self.available_modes():
            return True
        return False

    def broadcast_time(self):
        """repeat & broadcast time tensor to same shape as continuous/discrete
        (B,1) -> (B,D,1)
        """
        D = self.shape[-1]
        self.time = self.time.unsqueeze(1).repeat(1, D, 1)

    def apply_mask(self, condition=None):
        if "time" in self.available_modes():
            self.time *= self.mask if condition is None else condition
        if "continuous" in self.available_modes():
            self.continuous *= self.mask if condition is None else condition
        if "discrete" in self.available_modes():
            self.discrete = (
                (self.discrete * self.mask).long()
                if condition is None
                else (self.discrete * condition).long()
            )

    @classmethod
    def load_from(cls, path, device=None, transform=None) -> "TensorMultiModal":
        """
        Load tensors from an HDF5 file and optionally apply transformations.

        Args:
            path (str): Path to the HDF5 file.
            device (str, optional): Device to move tensors to.
            transform (callable or dict, optional): A callable function or dictionary of callables
                to apply transformations to specific tensors. Keys can be "time", "continuous",
                "discrete", or "mask".

        Returns:
            TensorMultiModal: An instance of the class with loaded and transformed tensors.
        """
        tensors = {"time": None, "continuous": None, "discrete": None, "mask": None}

        with h5py.File(path, "r") as f:
            for key in tensors.keys():
                if key in f:
                    tensors[key] = torch.from_numpy(f[key][:])

        if transform:
            if callable(transform):
                # Apply the same transform to all tensors
                tensors = {
                    key: (transform(tensor) if tensor is not None else None)
                    for key, tensor in tensors.items()
                }

            elif isinstance(transform, dict):
                # Apply specific transforms to individual tensors
                for key, func in transform.items():
                    if key in tensors and tensors[key] is not None and callable(func):
                        tensors[key] = func(tensors[key])

        state = cls(
            tensors["time"], tensors["continuous"], tensors["discrete"], tensors["mask"]
        )
        return state.to(device) if device else state

    @classmethod
    def sample_from(cls, distribution, shape, device=None) -> "TensorMultiModal":
        time, continuous, discrete, mask = distribution.sample(shape)
        state = cls(time, continuous, discrete, mask)
        if device:
            return state.to(device)
        return state

    @staticmethod
    def cat(states: List["TensorMultiModal"], dim=0) -> "TensorMultiModal":
        def cat_attr(attr_name):
            attrs = [getattr(s, attr_name, None) for s in states]
            if all(a is None for a in attrs):
                return None
            attrs = [a for a in attrs if a is not None]
            return torch.cat(attrs, dim=dim)

        return TensorMultiModal(
            time=cat_attr("time"),
            continuous=cat_attr("continuous"),
            discrete=cat_attr("discrete"),
            mask=cat_attr("mask"),
        )

    @staticmethod
    def stack(states: List["TensorMultiModal"], dim=0) -> "TensorMultiModal":
        def stack_attr(attr_name):
            attrs = [getattr(s, attr_name, None) for s in states]
            if all(a is None for a in attrs):
                return None
            attrs = [a for a in attrs if a is not None]
            return torch.stack(attrs, dim=dim)

        return TensorMultiModal(
            time=stack_attr("time"),
            continuous=stack_attr("continuous"),
            discrete=stack_attr("discrete"),
            mask=stack_attr("mask"),
        )

    def available_modes(
        self,
    ) -> List[str]:
        """Return a list of non-None modes in the state."""
        available_modes = []
        if self.time is not None:
            available_modes.append("time")
        if self.continuous is not None:
            available_modes.append("continuous")
        if self.discrete is not None:
            available_modes.append("discrete")
        return available_modes

    def save_to(self, path):
        with h5py.File(path, "w") as f:
            for mode in self.available_modes():
                f.create_dataset(mode, data=getattr(self, mode))
            f.create_dataset("mask", data=self.mask)

    def _apply_fn(self, func):
        """apply transformation function to all attributes."""
        return TensorMultiModal(
            time=func(self.time) if isinstance(self.time, torch.Tensor) else None,
            continuous=func(self.continuous)
            if isinstance(self.continuous, torch.Tensor)
            else None,
            discrete=func(self.discrete)
            if isinstance(self.discrete, torch.Tensor)
            else None,
            mask=func(self.mask) if isinstance(self.mask, torch.Tensor) else None,
        )

    def _apply_tensor_op(
        self, op, *args, mode: str = None, **kwargs
    ) -> "TensorMultiModal":
        """
        Apply a tensor operation to all or a specific mode.

        Args:
            op (callable): A tensor function (e.g., torch.unsqueeze, torch.reshape, torch.expand, torch.repeat).
            *args: Arguments to pass to the tensor operation.
            mode (str, optional): If provided, applies operation only to this mode.
            **kwargs: Keyword arguments for the tensor operation.

        Returns:
            TensorMultiModal: A new instance with the operation applied.
        """
        if mode is not None:
            if mode not in ["time", "continuous", "discrete", "mask"]:
                raise ValueError(
                    f"Invalid mode '{mode}'. Choose from ['time', 'continuous', 'discrete', 'mask']"
                )

        return TensorMultiModal(
            time=op(self.time, *args, **kwargs)
            if self.time is not None and (mode is None or mode == "time")
            else self.time,
            continuous=op(self.continuous, *args, **kwargs)
            if self.continuous is not None and (mode is None or mode == "continuous")
            else self.continuous,
            discrete=op(self.discrete, *args, **kwargs)
            if self.discrete is not None and (mode is None or mode == "discrete")
            else self.discrete,
            mask=op(self.mask, *args, **kwargs)
            if self.mask is not None and (mode is None or mode == "mask")
            else self.mask,
        )
