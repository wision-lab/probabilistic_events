import torch
from jaxtyping import Float
from loguru import logger
from torch import nn, Tensor
from torch.amp import autocast
from torch.nn import functional as F
from tqdm import tqdm

# Assuming timer utilities are in this path
from probabilistic_events.utils.timer import CudaTimer, CPUTimer


class AdaptiveWienerFilter(nn.Module):
    """
    Applies a counts-aware adaptive Wiener filter for spatial denoising
    using a separable Gaussian kernel for local statistics.

    This module caches the internal 1D filter kernels for efficient,
    repeated application (e.g., during inference).

    It seamlessly handles both grayscale (H, W) and multi-channel (C, H, W)
    probability maps, expecting a single (H, W) counts map.
    """

    def __init__(
        self,
        window_size: int = 7,
        noise_gain: float = 0.8,
        sigma: float | None = None,
    ):
        super().__init__()
        assert window_size % 2 == 1, "window_size must be odd for Gaussian kernel"
        self.window_size = window_size
        self.noise_gain = noise_gain
        self.epsilon = 1e-9

        if sigma is None:
            # Standard torchvision heuristic for sigma from window_size
            self.sigma = 0.3 * ((window_size - 1) * 0.5 - 1) + 0.8
        else:
            self.sigma = sigma

        self.register_buffer("_cached_kernel_v", None, persistent=False)
        self.register_buffer("_cached_kernel_h", None, persistent=False)

    def train(self, mode: bool = True):
        """Override train to invalidate the cache."""
        super().train(mode)
        self._cached_kernel_v = None
        self._cached_kernel_h = None
        return self

    def eval(self):
        """Override eval to invalidate the cache."""
        super().eval()
        self._cached_kernel_v = None
        self._cached_kernel_h = None
        return self

    def _build_and_cache_gaussian_kernels(
        self, device: torch.device, dtype: torch.dtype
    ):
        """Internal method to compute and cache the 1D Gaussian kernels."""
        t = (
            torch.arange(self.window_size, device=device, dtype=dtype)
            - self.window_size // 2
        )
        kernel_1d = torch.exp(-0.5 * (t / self.sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        self._cached_kernel_v = kernel_1d
        self._cached_kernel_h = kernel_1d

    @staticmethod
    def _separable_conv(
        image: Float[Tensor, "batch channel height width"],
        kernel_v: Float[Tensor, "k_v"],
        kernel_h: Float[Tensor, "k_h"],
    ) -> Float[Tensor, "batch channel height width"]:
        """
        Performs a 2D separable convolution using two 1D kernels via F.conv2d.
        Applies the convolution to each channel in `image` independently.
        """
        n, c, h, w = image.shape
        k_size_v = kernel_v.shape[0]
        k_size_h = kernel_h.shape[0]
        pad_v = k_size_v // 2
        pad_h = k_size_h // 2

        dev, dt = image.device, image.dtype

        k_v = (
            kernel_v.to(device=dev, dtype=dt).view(1, 1, k_size_v, 1).repeat(c, 1, 1, 1)
        )
        k_h = (
            kernel_h.to(device=dev, dtype=dt).view(1, 1, 1, k_size_h).repeat(c, 1, 1, 1)
        )

        y = F.conv2d(image, k_v, padding=(pad_v, 0), groups=c)
        y = F.conv2d(y, k_h, padding=(0, pad_h), groups=c)
        return y

    def forward(
        self,
        probabilities: Float[Tensor, "... height width"],
        counts: Float[Tensor, "height width"],
    ) -> Float[Tensor, "... height width"]:
        """
        Applies the adaptive Wiener filter.

        Args:
            probabilities: The temporally fused probability map.
                           Can be (H, W) for grayscale or (C, H, W) for color.
            counts: The temporally fused counts map (H, W).
                    This will be broadcast across channels.

        Returns:
            The spatially denoised probability map, with the same
            dimensions as the input `probabilities`.
        """
        device = probabilities.device
        compute_dtype = torch.float32
        input_dtype = probabilities.dtype
        orig_ndim = probabilities.ndim

        cache_valid = (
            not self.training
            and self._cached_kernel_v is not None
            and self._cached_kernel_h is not None
            and self._cached_kernel_v.device == device
            and self._cached_kernel_v.dtype == compute_dtype
            and self._cached_kernel_h.device == device
            and self._cached_kernel_h.dtype == compute_dtype
        )

        if not cache_valid:
            self._build_and_cache_gaussian_kernels(device, compute_dtype)

        kernel_v = self._cached_kernel_v
        kernel_h = self._cached_kernel_h

        probs_compute = probabilities.to(compute_dtype)
        counts_compute = counts.to(compute_dtype)

        # Reshape to [B, C, H, W] for conv
        if orig_ndim == 2:
            # Input is (H, W) -> (1, 1, H, W)
            probs_bchw = probs_compute.unsqueeze(0).unsqueeze(0)
        elif orig_ndim == 3:
            # Input is (C, H, W) -> (1, C, H, W)
            probs_bchw = probs_compute.unsqueeze(0)
        else:
            raise ValueError(
                f"Input probabilities must be 2D (H, W) or 3D (C, H, W), "
                f"got {orig_ndim}D."
            )

        # Reshape counts to (1, 1, H, W) for broadcasting
        counts_bchw = counts_compute.unsqueeze(0).unsqueeze(0)

        # Compute local mean
        local_mean_bchw = self._separable_conv(probs_bchw, kernel_v, kernel_h)

        # Compute local mean of squares
        fused_prob_sq_bchw = probs_bchw.pow(2)
        local_mean_sq_bchw = self._separable_conv(
            fused_prob_sq_bchw, kernel_v, kernel_h
        )

        # E[P^2] - (E[P])^2
        local_var_bchw = local_mean_sq_bchw - local_mean_bchw.pow(2)

        # Estimate noise variance.
        # (probs_bchw * (1 - probs_bchw)) is [1, C, H, W]
        # (counts_bchw + self.epsilon) is [1, 1, H, W]
        # The division broadcasts correctly.
        noise_var_bchw = (
            self.noise_gain
            * (probs_bchw * (1 - probs_bchw))
            / (counts_bchw + self.epsilon)
        )

        # Compute the filter
        variance_ratio = torch.clamp(local_var_bchw - noise_var_bchw, min=0) / (
            local_var_bchw + self.epsilon
        )
        denoised_prob_bchw = local_mean_bchw + variance_ratio * (
            probs_bchw - local_mean_bchw
        )

        # Clamp and cast back to original input shape
        final_denoised = torch.clamp(denoised_prob_bchw, 0, 1)

        if orig_ndim == 2:
            return final_denoised.squeeze(0).squeeze(0).to(input_dtype)
        else:  # orig_ndim == 3
            return final_denoised.squeeze(0).to(input_dtype)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    Timer = CudaTimer if device.type == "cuda" else CPUTimer

    # Test data
    probabilities_mono = torch.rand(256, 512).to(device)
    probabilities_color = torch.rand(3, 256, 512).to(device)
    counts_map = torch.rand(256, 512).to(device) * 100 + 1

    warm_iters = 100
    eval_iter = 10_000

    wiener_filter = AdaptiveWienerFilter(window_size=7, noise_gain=0.8)
    wiener_filter = wiener_filter.to(device).eval()
    compiled_wiener_filter = torch.compile(wiener_filter)

    # --- Test Grayscale ---
    logger.info("Warming up (Grayscale)...")
    for _ in range(warm_iters):
        _ = compiled_wiener_filter(probabilities_mono, counts_map)

    logger.info(f"Running benchmark (Grayscale, {eval_iter} iterations)...")
    with Timer() as t_instance_mono, autocast(device_type=device.type):
        for _ in tqdm(range(eval_iter)):
            _ = compiled_wiener_filter(probabilities_mono, counts_map)

    logger.info(
        f"Grayscale Iter time: {float(t_instance_mono) / eval_iter:.3g} sec. "
        f"Throughput: {eval_iter / float(t_instance_mono):.3g} FPS"
    )

    # --- Test Color ---
    logger.info("Warming up (Color)...")
    for _ in range(warm_iters):
        _ = compiled_wiener_filter(probabilities_color, counts_map)

    logger.info(f"Running benchmark (Color, {eval_iter} iterations)...")
    with Timer() as t_instance_color, autocast(device_type=device.type):
        for _ in tqdm(range(eval_iter)):
            _ = compiled_wiener_filter(probabilities_color, counts_map)

    logger.info(
        f"Color Iter time: {float(t_instance_color) / eval_iter:.3g} sec. "
        f"Throughput: {eval_iter / float(t_instance_color):.3g} FPS"
    )
