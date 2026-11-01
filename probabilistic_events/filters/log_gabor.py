import math
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage import data
from skimage.transform import resize
from torch import Tensor

# Assumed dependencies
from probabilistic_events.ops.array_ops import binomial_vst
from probabilistic_events.utils.timer import CudaTimer, CPUTimer


class LogGabor(nn.Module):
    """
    Log-Gabor Filter Bank with analytical covariance calculation.
    """

    def __init__(
        self,
        num_scales: int = 4,
        num_orientations: int = 6,
        min_wavelength: float = 3.0,
        scale_multiplier: float = 2.0,
        sigma_on_f: float = 0.55,
        ratio_angular_sigma_on_theta: float = 0.5,
        standardize: bool = True,
        fft_artifact_handling: Literal["none", "padding"] = "padding",
        subspace_components: Optional[int] = None,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.min_wavelength = min_wavelength
        self.scale_multiplier = scale_multiplier
        self.sigma_on_f = sigma_on_f
        self.ratio_angular_sigma_on_theta = ratio_angular_sigma_on_theta
        self.standardize = standardize
        self.fft_artifact_handling = fft_artifact_handling
        self.subspace_components = subspace_components

        self.num_bandpass_filters = num_scales * num_orientations
        self.num_features = self.num_bandpass_filters + 1
        if self.subspace_components is not None:
            if self.subspace_components > self.num_features:
                self.subspace_components = self.num_features

        self.register_buffer("is_fitted", torch.tensor(False, dtype=torch.bool))

        # PCA buffers
        self.register_buffer(
            "_cached_eigenvectors",
            torch.zeros(self.num_features, self.num_features),
            persistent=False,
        )
        self.register_buffer(
            "_cached_eigenvalues",
            torch.zeros(self.num_features),
            persistent=False,
        )

        # Grid caching
        self.register_buffer("_cached_height", torch.tensor(0), persistent=False)
        self.register_buffer("_cached_width", torch.tensor(0), persistent=False)
        self.register_buffer("_grid_radius", torch.tensor([]), persistent=False)
        self.register_buffer("_grid_theta", torch.tensor([]), persistent=False)

    def _precompute_grid(self, height: int, width: int, device: torch.device):
        if (
            self._cached_height == height
            and self._cached_width == width
            and self._grid_radius.numel() > 0
        ):
            return

        y_coords = torch.fft.fftfreq(height, device=device)
        x_coords = torch.fft.fftfreq(width, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        radius = torch.sqrt(grid_x**2 + grid_y**2)
        theta = torch.atan2(-grid_y, grid_x)

        # Avoid log(0) at DC component
        radius[0, 0] = 1.0

        self._grid_radius = radius
        self._grid_theta = theta
        self._cached_height.fill_(height)
        self._cached_width.fill_(width)

    def _build_filters(self, height: int, width: int, device: torch.device) -> Tensor:
        self._precompute_grid(height, width, device)
        radius = self._grid_radius
        theta = self._grid_theta

        filters = []
        log_radius = torch.log(radius)
        y_indices = torch.arange(height, device=device)
        x_indices = torch.arange(width, device=device)

        for scale_index in range(self.num_scales):
            orientation_shift = (
                0.5 * (math.pi / self.num_orientations) if scale_index % 2 == 1 else 0.0
            )
            current_wavelength = self.min_wavelength * (
                self.scale_multiplier**scale_index
            )
            center_frequency = 1.0 / current_wavelength
            log_center_frequency = math.log(center_frequency)
            radial_component = torch.exp(
                -((log_radius - log_center_frequency) ** 2)
                / (2 * (math.log(self.sigma_on_f) ** 2))
            )
            radial_component[0, 0] = 0.0

            for orientation_index in range(self.num_orientations):
                center_angle = (
                    orientation_index * (math.pi / self.num_orientations)
                    + orientation_shift
                )

                delta_theta = torch.remainder(
                    theta - center_angle + 0.5 * math.pi,
                    math.pi,
                )
                delta_theta = torch.abs(delta_theta - 0.5 * math.pi)

                sigma_theta = (
                    math.pi
                    / self.num_orientations
                    / (self.ratio_angular_sigma_on_theta * 2.0)
                )
                angular_component = torch.exp(-(delta_theta**2) / (2 * sigma_theta**2))

                transfer_function = radial_component * angular_component
                transfer_function = 0.5 * (
                    transfer_function
                    + transfer_function[(-y_indices) % height][:, (-x_indices) % width]
                )
                filters.append(transfer_function)

        filter_stack = torch.stack(filters)
        band_energy = filter_stack.square().sum(dim=0)
        lowpass_filter = (1.0 - band_energy).clamp_min(0.0).sqrt()
        lowpass_filter[0, 0] = 1.0
        filter_stack = torch.cat([filter_stack, lowpass_filter.unsqueeze(0)], dim=0)
        frame_denominator = filter_stack.square().sum(dim=0).clamp_min(1.0e-6)
        filter_stack = filter_stack / frame_denominator.sqrt().unsqueeze(0)

        return filter_stack

    def _build_rfft_filters(
        self, height: int, width: int, device: torch.device
    ) -> Tensor:
        """Return the real-valued half-spectrum filters used by ``rfft2``.

        The bank depends only on the filter parameters and the frame size, so it
        is built once per (size, device) and reused; rebuilding it every call
        dominated the runtime at small resolutions.
        """
        key = (height, width, str(device))
        cache = self.__dict__.setdefault("_rfft_filter_cache", {})
        if key not in cache:
            cache[key] = self._build_filters(height, width, device)[..., : width // 2 + 1].contiguous()
        return cache[key]

    @torch.jit.export
    def fit_subspace(self):
        """
        Computes analytical covariance matrix and eigendecomposition in Frequency Domain.
        Automatically fuses whitening scaling into the eigenvectors.
        """
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = next(self.buffers()).device

        calculation_height, calculation_width = 128, 128

        if self.fft_artifact_handling == "padding":
            max_scale_factor = self.scale_multiplier ** (self.num_scales - 1)
            padding_margin = int(self.min_wavelength * max_scale_factor)
            calculation_height += 2 * padding_margin
            calculation_width += 2 * padding_margin

        filters = self._build_filters(calculation_height, calculation_width, device)

        num_filters = filters.shape[0]
        filters_flat = filters.view(num_filters, -1)

        # Compute Covariance Matrix (Real approximation via Parseval's)
        covariance_matrix = (filters_flat @ filters_flat.T) / (
            calculation_height * calculation_width
        )

        # Regularization for numerical stability
        covariance_matrix = (
            covariance_matrix + torch.eye(num_filters, device=device) * 1e-4
        )

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
        sorted_indices = torch.argsort(eigenvalues, descending=True)

        sorted_eigenvalues = eigenvalues[sorted_indices].to(torch.float32)
        sorted_eigenvectors = eigenvectors[:, sorted_indices].to(torch.float32)

        # Fuse whitening into the eigenvectors. Full-rank PCA includes the
        # smallest covariance modes, so floor eigenvalues relative to the
        # spectrum before rsqrt to avoid amplifying numerical null directions.
        eigenvalue_floor = torch.clamp(sorted_eigenvalues[0] * 1.0e-3, min=1e-6)
        safe_eigenvalues = sorted_eigenvalues.clamp_min(eigenvalue_floor)
        whitening_scale = torch.rsqrt(safe_eigenvalues)
        whitened_vectors = sorted_eigenvectors * whitening_scale.unsqueeze(0)
        whitened_vectors = torch.nan_to_num(whitened_vectors, nan=0.0)

        self._cached_eigenvalues = sorted_eigenvalues
        self._cached_eigenvectors = whitened_vectors
        self.is_fitted.fill_(True)

    @torch.compiler.disable
    def forward(
        self, image_input: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, "batch features height width"]:

        batch_size, channels, height, width = image_input.shape
        device = image_input.device

        pad_height, pad_width = 0, 0
        if self.fft_artifact_handling == "padding":
            max_sigma = (
                self.min_wavelength
                * (self.scale_multiplier ** (self.num_scales - 1))
                * 0.6
            )
            padding_size = int(math.ceil(3 * max_sigma))
            pad_height, pad_width = padding_size, padding_size

        effective_height = height + 2 * pad_height
        effective_width = width + 2 * pad_width

        if pad_height > 0:
            padded_image = F.pad(
                image_input,
                (pad_width, pad_width, pad_height, pad_height),
                mode="reflect",
            )
        else:
            padded_image = image_input

        image_fft = torch.fft.rfft2(padded_image)
        filters = self._build_rfft_filters(effective_height, effective_width, device)

        # Real-valued quadrature-free filtering on the Hermitian half-spectrum.
        output_fft = image_fft * filters.unsqueeze(0)
        output_spatial = torch.fft.irfft2(
            output_fft,
            s=(effective_height, effective_width),
        )

        if pad_height > 0:
            output_spatial = output_spatial[
                ..., pad_height:-pad_height, pad_width:-pad_width
            ]

        return output_spatial

    @torch.compiler.disable
    def forward_projected(
        self, image_input: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, "batch subspace_k height width"]:

        if not self.is_fitted:
            raise RuntimeError("Subspace not fitted. Call .fit_subspace() first.")

        features = self.forward(image_input)

        # Use the already whitened eigenvectors
        projection_matrix = self._cached_eigenvectors
        if self.subspace_components is not None:
            projection_matrix = projection_matrix[:, : self.subspace_components]

        projection_matrix = projection_matrix.to(features.dtype)

        return torch.einsum("bkhw, ks -> bshw", features, projection_matrix)


def run_loggabor_response_analysis(
    module: "LogGabor",
    batch_images: Tensor,
    num_scales: int,
    num_orientations: int,
    sum_frames: int,
    pdf_path: Path,
):
    logger.info("Running LogGabor response analysis...")
    responses = module(batch_images)
    responses_cpu = responses.cpu().detach().numpy()

    batch_size, _, height, width = responses_cpu.shape

    # Reshape: [Batch, Orientations, Scales, Height, Width]
    # Note: Filter construction loop was Orientation (outer) -> Scale (inner)
    responses_reshaped = responses_cpu.reshape(
        batch_size, num_orientations, num_scales, height, width
    )

    # Transpose to [Batch, Scale, Orientation, Height, Width]
    responses_reshaped = responses_reshaped.transpose(0, 2, 1, 3, 4)

    with PdfPages(pdf_path) as pdf:
        labels = ["VST(Original)", f"VST(Binomial Sum={sum_frames})"]
        for img_idx, img_label in enumerate(labels):
            fig, axes = plt.subplots(
                num_scales,
                num_orientations,
                figsize=(3 * num_orientations, 3 * num_scales),
                sharex=True,
                sharey=True,
                layout="constrained",
            )
            fig.suptitle(f"LogGabor Responses ({img_label})", fontsize=16)

            for scale_idx in range(num_scales):
                for orient_idx in range(num_orientations):
                    response_map = responses_reshaped[img_idx, scale_idx, orient_idx]
                    abs_max = np.abs(response_map).max()

                    if num_scales > 1 and num_orientations > 1:
                        ax = axes[scale_idx, orient_idx]
                    elif num_orientations > 1:
                        ax = axes[orient_idx]
                    else:
                        ax = axes[scale_idx]

                    im = ax.imshow(
                        response_map, cmap="coolwarm", vmin=-abs_max, vmax=abs_max
                    )

                    if scale_idx == 0:
                        ax.set_title(f"Orient {orient_idx*30}°")
                    if orient_idx == 0:
                        ax.set_ylabel(f"Scale {scale_idx}")

            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
            pdf.savefig(fig)
            plt.close(fig)

    logger.info(f"Saved LogGabor response plots to '{pdf_path.resolve()}'")


def run_loggabor_pca_analysis(module: "LogGabor", pdf_path: Path):
    logger.info("Plotting PCA explained variance...")

    eigenvalues_cpu = module._cached_eigenvalues.cpu().to(torch.float32).numpy()
    num_components = len(eigenvalues_cpu)

    explained_variance = eigenvalues_cpu / eigenvalues_cpu.sum()
    cumulative_variance = np.cumsum(explained_variance)

    components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    components_99 = np.argmax(cumulative_variance >= 0.99) + 1

    logger.info(
        f"PCA Report: 95% Var = {components_95} comps, 99% Var = {components_99} comps"
    )

    with PdfPages(pdf_path) as pdf:
        fig, ax1 = plt.subplots(figsize=(12, 7), layout="constrained")
        fig.suptitle("PCA Explained Variance for LogGabor", fontsize=16)

        color_bar = "tab:blue"
        ax1.set_xlabel("Component")
        ax1.set_ylabel("Explained Variance", color=color_bar)
        ax1.bar(range(num_components), explained_variance, color=color_bar, alpha=0.6)

        ax2 = ax1.twinx()
        color_line = "tab:red"
        ax2.set_ylabel("Cumulative Variance", color=color_line)
        ax2.plot(
            range(num_components), cumulative_variance, color=color_line, marker="."
        )

        ax2.axhline(
            y=0.99, color="orange", linestyle=":", label=f"99% ({components_99})"
        )
        ax2.axhline(
            y=0.95, color="green", linestyle=":", label=f"95% ({components_95})"
        )
        fig.legend(loc="center right")

        pdf.savefig(fig)
        plt.close(fig)

    logger.info(f"PCA plot saved to '{pdf_path.resolve()}'")


def run_loggabor_benchmark(module: "LogGabor", batch_images: Tensor, pdf_path: Path):
    logger.info("--- Starting Performance Benchmark ---")
    batch_sizes = [1, 2, 4, 8]
    timing_results = []

    timer_cls = CudaTimer if batch_images.device.type == "cuda" else CPUTimer

    for size in batch_sizes:
        test_batch = batch_images[0:1].repeat(size, 1, 1, 1)

        # Warmup
        _ = module.forward_projected(test_batch)

        with timer_cls() as timer_context:
            _ = module.forward_projected(test_batch)

        elapsed_seconds = float(timer_context)
        timing_results.append(elapsed_seconds)

        throughput = size / elapsed_seconds if elapsed_seconds > 0 else 0
        logger.info(f"Batch {size}: {elapsed_seconds:.4f}s | {throughput:.2f} img/s")

    with PdfPages(pdf_path) as pdf:
        fig, ax1 = plt.subplots(figsize=(8, 5), layout="constrained")
        fig.suptitle("LogGabor Inference Performance (FFT)", fontsize=16)
        ax1.plot(batch_sizes, timing_results, "o-", label="Time")
        ax1.set_xticks(batch_sizes)
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Time (s)")
        pdf.savefig(fig)
        plt.close(fig)

    logger.info(f"Saved benchmark to '{pdf_path.resolve()}'")


if __name__ == "__main__":
    # 1. Configuration
    num_scales = 4
    num_orientations = 6
    sum_frames = 32
    max_ppp = 1
    image_size = (256, 256)
    standardize = True

    # 2. Setup
    output_folder = Path("outputs/log_gabor")
    output_folder.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 3. Data Preparation
    original_image_np = data.camera().astype(np.float32) / 255.0
    original_image = torch.from_numpy(
        resize(original_image_np, image_size, anti_aliasing=True)
    ).to(device)

    probability_map = original_image * max_ppp / sum_frames

    # Simulate binomial noise
    frame_stack = torch.rand(
        sum_frames, *image_size, device=device
    ) < probability_map.unsqueeze(0)
    noisy_image_counts = frame_stack.sum(dim=0).float()

    noisy_image_vst = binomial_vst(noisy_image_counts, sum_frames)
    original_image_vst = binomial_vst(original_image * sum_frames, sum_frames)

    batch_images = torch.stack([original_image_vst, noisy_image_vst]).unsqueeze(1)

    # 4. Initialize Module & Fit Subspace
    module = (
        LogGabor(
            num_scales=num_scales,
            num_orientations=num_orientations,
            standardize=standardize,
            subspace_components=None,
        )
        .to(device)
        .eval()
    )

    module.fit_subspace()

    # 5. Run Analyses
    run_loggabor_response_analysis(
        module,
        batch_images,
        num_scales,
        num_orientations,
        sum_frames,
        output_folder / "loggabor_responses.pdf",
    )

    run_loggabor_pca_analysis(module, output_folder / "loggabor_pca.pdf")

    run_loggabor_benchmark(module, batch_images, output_folder / "loggabor_perf.pdf")

    logger.info("--- All LogGabor Analyses Complete ---")
