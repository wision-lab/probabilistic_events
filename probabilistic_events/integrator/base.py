from abc import ABC, abstractmethod

import torch
from einops import rearrange
from jaxtyping import Float
from torch import nn, Tensor
from torch.nn import functional as F


class BaseMotionMetering(nn.Module, ABC):
    """
    An abstract base class containing the shared logic for motion metering integrators.
    """

    def __init__(
        self,
        bocpd_gamma: float,
        sum_frames: int,
        memory_size: int,
        min_filter_size: int,
        **kwargs,  # Absorb unused kwargs from subclasses
    ):
        super().__init__()
        assert min_filter_size % 2 == 1, "Min. filter size must be odd"

        # Per-frame hazard -> probability of >=1 change within sum_frames frames
        self.bocpd_gamma = 1 - (1 - bocpd_gamma) ** sum_frames
        self.sum_frames = sum_frames
        self.memory_size = memory_size
        self.max_filter_size = min_filter_size

        # Internal state variables
        self._eps = 1e-20
        self.t_absolute = 0
        self.log_forecaster_distribution_ll = None
        self.forecaster_distribution_index_ll = None
        self.recons = None

    def reset(self):
        """Resets the internal time state of the meter for processing a new sequence."""
        self.t_absolute = 0
        self.log_forecaster_distribution_ll = None

    def max_filter_2d(self, x: Float[Tensor, "h w"]) -> Float[Tensor, "h w"]:
        """Performs 2D max-pooling on an image."""
        x_batched = rearrange(x, "h w -> 1 1 h w")
        pooled = F.max_pool2d(
            x_batched,
            kernel_size=self.max_filter_size,
            stride=1,
            padding=(self.max_filter_size - 1) // 2,
        )
        return rearrange(pooled, "1 1 h w -> h w")

    @property
    def forecaster_distribution_ll(self) -> Float[Tensor, "h w"]:
        return torch.softmax(self.log_forecaster_distribution_ll, dim=-1)

    def estimate_sample_weight(
        self,
    ) -> Float[Tensor, "h w"]:
        """
        Estimates the sample weight based on the BOCPD run-length distribution.
        Uses the harmonic mean (1/n) to approximate a restartable arithmetic mean.
        """
        # (self.t_absolute - forecaster_indices) gives run length in binary frames.
        # Dividing by self.sum_frames converts it to slow-clock steps.
        run_lengths_in_steps = (
            self.t_absolute - self.forecaster_distribution_index_ll
        ).clamp(min=1.0) / self.sum_frames

        # LINEAR / HARMONIC FORMULA
        # This ensures that at run_length=1, the weight is 1.0 (hard reset).
        # sample_weights = 1.0 / run_lengths_in_steps.clamp(min=1.0)
        sample_weights = 2.0 / (run_lengths_in_steps.clamp(min=1.0) + 1)


        return (self.forecaster_distribution_ll * sample_weights).sum(dim=-1)

    def estimate_run_length(
        self,
    ) -> Float[Tensor, "h w"]:
        """
        Estimates the run length based on the BOCPD run-length distribution.

        (In binary frame indices)
        """
        run_lengths = self.t_absolute - self.forecaster_distribution_index_ll

        return (self.forecaster_distribution_ll * run_lengths).sum(dim=-1)

    @abstractmethod
    def forward(self, photon_cube: Float[Tensor, "h w t"]):
        """Subclasses must implement their own forward pass."""
        raise NotImplementedError


def log_binom(n: Tensor | int, k: Tensor) -> Tensor:
    """Computes the log of the binomial coefficient."""
    if isinstance(n, int):
        n = torch.tensor(n, device=k.device, dtype=k.dtype)

    return torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)


def log_beta(a: Tensor, b: Tensor) -> Tensor:
    """Computes the log of the Beta function."""
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)


def update_tensor(
    tensor: Tensor, index: Tensor, value: float | Tensor, mask: Tensor, dim: int = -1
) -> Tensor:
    """
    Conditionally updates a tensor along a specified dimension at indices
    given by `index`, based on a boolean mask.

    This function is generalized to handle tensors with any number of leading
    dimensions by leveraging PyTorch's broadcasting.

    :param tensor: The tensor to update in-place.
    :param index: A tensor of indices for the update. Its shape must be
                  broadcastable to the shape of `tensor` excluding the `dim` dimension.
    :param value: The new value to insert. Can be a scalar or a tensor
                  broadcastable to the shape of `index`.
    :param mask: A boolean tensor with the same shape as `index`. The update is
                 only performed where this mask is True.
    :param dim: The dimension along which to update. Defaults to the last dimension.
    :return: The updated tensor.
    """
    index_expanded = index.unsqueeze(dim)

    original_value = tensor.gather(dim, index_expanded).squeeze(dim)

    updated_value = torch.where(mask, value, original_value)

    tensor.scatter_(dim, index_expanded, updated_value.unsqueeze(dim))

    return tensor
