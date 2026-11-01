from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch import Tensor, nn
from torch.nn import functional as F


def separable_conv(
    image: Tensor,
    kernel_v: Tensor,
    kernel_h: Tensor,
) -> Tensor:
    """
    Performs a 2D separable convolution using two 1D kernels via F.conv2d.
    Applies the convolution to each channel in `image` independently.

    Optimized to use broadcasting and avoid unnecessary device transfers.
    """
    n, c, h, w = image.shape

    # Reshape kernels for depthwise convolution: (Out, In/Groups, kH, kW)
    # Groups = C, so In/Groups = 1. Out = C.
    # We expand the 1D kernels to (C, 1, K, 1) and (C, 1, 1, K)

    # Note: We assume kernels are already on the correct device/dtype
    k_v = kernel_v.view(1, 1, -1, 1).expand(c, -1, -1, -1)
    k_h = kernel_h.view(1, 1, 1, -1).expand(c, -1, -1, -1)

    pad_v = kernel_v.shape[0] // 2
    pad_h = kernel_h.shape[0] // 2

    # Vertical pass
    y = F.conv2d(image, k_v, padding=(pad_v, 0), groups=c)
    # Horizontal pass
    y = F.conv2d(y, k_h, padding=(0, pad_h), groups=c)

    return y


@torch.compile()
class FaridSimoncelli(nn.Module):
    """
    An nn.Module to compute and cache Farid & Simoncelli derivative filters.

    This module computes the FS04 filters once during initialization and
    registers them as buffers.

    If standardize=True, the filters are normalized such that the output variance
    is 1.0 when the input is unit variance white noise.
    """

    def __init__(self, num_tap: int = 5, num_order: int = 1, standardize: bool = True):
        """
        :param num_tap: The number of taps for the filter (3, 5, or 7).
        :param num_order: The order of the derivative (1 or 2).
        :param standardize: If True, normalizes filter outputs by their energy (Whitening).
        """
        super().__init__()
        self.num_tap = num_tap
        self.num_order = num_order
        self.standardize = standardize

        p, d1, d2 = self.fs04_kernels(num_tap=num_tap, num_order=num_order)

        self.register_buffer("p_filter", p.float())
        self.register_buffer("d1_filter", d1.float())

        # Compute energy normalization
        # Note: We compute energy based on the full 2D separable kernel
        energy_d1 = self.fs04_energy(p, d1, None)
        self.register_buffer("norm_d1", torch.sqrt(energy_d1).clamp(min=1e-8).float())

        if d2 is not None and num_order == 2:
            self.register_buffer("d2_filter", d2.float())
            _, energy_d2 = self.fs04_energy(p, d1, d2)
            self.register_buffer(
                "norm_d2", torch.sqrt(energy_d2).clamp(min=1e-8).float()
            )
        else:
            self.d2_filter = None
            self.norm_d2 = None

    @staticmethod
    def fs04_kernels(
        num_tap: int = 3, num_order: int = 1
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Generates Farid and Simoncelli's discrete differentiation filters.
        Correctly constructs symmetric/antisymmetric kernels from coefficients.

        :param num_tap: The number of taps for the filter (3, 5, or 7).
        :param num_order: The order of the derivative (1 or 2).
        :return: A tuple of (pre_filter, first_order_deriv, second_order_deriv).
        """
        # Coefficients are provided as [Center, Edge1, Edge2...]
        if num_tap == 3:
            p = torch.tensor([0.540242, 0.229879])
            d1 = torch.tensor([-0.425287])
            d2 = None
        elif num_tap == 5:
            p = torch.tensor([0.426375, 0.249153, 0.037659])
            d1 = torch.tensor([-0.276691, -0.109604])
            d2 = (
                torch.tensor([-0.471147, 0.002668, 0.232905])
                if num_order == 2
                else None
            )
        elif num_tap == 7:
            p = torch.tensor([0.361117, 0.245410, 0.069321, 0.004711])
            d1 = torch.tensor([-0.193091, -0.125376, -0.018708])
            d2 = (
                torch.tensor([-0.273118, -0.056554, 0.137778, 0.055336])
                if num_order == 2
                else None
            )
        else:
            raise ValueError("num_tap must be 3, 5, or 7")

        # Construct Symmetric Pre-filter
        # p input is [c, e1, e2].
        # Target is [e2, e1, c, e1, e2].
        # p[1:] is [e1, e2]. Flip is [e2, e1].
        pre_filter = torch.cat((p[1:].flip(0), p))

        # Construct Antisymmetric First Derivative
        # d1 input is [d1, d2].
        # Target is [-d2, -d1, 0, d1, d2]
        first_order_deriv = torch.cat((-d1.flip(0), torch.tensor([0.0]), d1))

        # Construct Symmetric Second Derivative
        second_order_deriv = None
        if d2 is not None:
            # d2 input is [c, e1, e2]
            second_order_deriv = torch.cat((d2[1:].flip(0), d2))

        return pre_filter, first_order_deriv, second_order_deriv

    @staticmethod
    def fs04_energy(
        p_filter: Tensor,
        d1_filter: Tensor,
        d2_filter: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Computes the total energy of the 2D separable kernels.
        Energy = Sum(Kernel_V^2) * Sum(Kernel_H^2)
        """
        sum_sq_p = torch.sum(p_filter**2)

        # Energy for Gx = P(y) * D1(x) -> SumSq(P) * SumSq(D1)
        energy_d1 = sum_sq_p * torch.sum(d1_filter**2)

        if d2_filter is not None:
            energy_d2 = sum_sq_p * torch.sum(d2_filter**2)
            return energy_d1, energy_d2

        return energy_d1

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        """
        Computes gradient responses using the cached FS04 kernels.

        :param image: The input image tensor (N, C, H, W).
        :return: A tuple of gradient images (grad_x, grad_y, ...).
        """
        p = self.p_filter
        d1 = self.d1_filter

        grad_x = separable_conv(image, p, d1)
        grad_y = separable_conv(image, d1, p)

        if self.standardize:
            grad_x = grad_x / self.norm_d1
            grad_y = grad_y / self.norm_d1

        if self.num_order == 1 or self.d2_filter is None:
            return grad_x, grad_y

        d2 = self.d2_filter
        grad_xx = separable_conv(image, p, d2)
        grad_yy = separable_conv(image, d2, p)
        grad_xy = separable_conv(grad_x, d1, p)

        if self.standardize:
            grad_xx = grad_xx / self.norm_d2
            grad_yy = grad_yy / self.norm_d2

        return grad_x, grad_y, grad_xx, grad_xy, grad_yy


def run_fs_covariance_analysis():
    logger.info("--- Running Farid-Simoncelli Covariance Analysis ---")

    NUM_TAP = 5
    NUM_ORDER = 1
    STANDARDIZE = True
    BATCH_SIZE = 16
    IMAGE_SIZE = (256, 256)

    OUTPUT_DIR = Path("outputs/farid_simoncelli")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputs will be saved to: {OUTPUT_DIR.resolve()}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    fs_filter = (
        FaridSimoncelli(num_tap=NUM_TAP, num_order=NUM_ORDER, standardize=STANDARDIZE)
        .to(DEVICE)
        .eval()
    )

    logger.info("Generating white noise input...")
    noise_images = torch.randn(
        BATCH_SIZE, 1, IMAGE_SIZE[0], IMAGE_SIZE[1], device=DEVICE
    )

    logger.info("Applying filter...")
    with torch.no_grad():
        grad_x, grad_y = fs_filter(noise_images)

    grad_x_flat = grad_x.reshape(1, -1)
    grad_y_flat = grad_y.reshape(1, -1)

    data = torch.cat([grad_x_flat, grad_y_flat], dim=0)
    logger.info(f"Data matrix shape (features, samples): {data.shape}")

    logger.info("Computing covariance matrix...")
    cov_matrix = torch.cov(data)
    cov_matrix_np = cov_matrix.detach().cpu().numpy()

    logger.info("Covariance Matrix (Gx, Gy):")
    logger.info(cov_matrix_np)

    # Check if diagonal is close to 1.0 (Whitened)
    diag_mean = np.mean(np.diag(cov_matrix_np))
    logger.info(f"Mean Variance (Target 1.0): {diag_mean:.4f}")

    logger.info("Plotting covariance matrix...")
    fig, ax = plt.subplots()
    im = ax.matshow(cov_matrix_np, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(im, label="Covariance")

    ax.set_title(
        f"Covariance: $G_x$ vs $G_y$ (on N(0,1) Noise)\nnum_tap={NUM_TAP}, standardize={STANDARDIZE}"
    )
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["$G_x$", "$G_y$"])
    ax.set_yticklabels(["$G_x$", "$G_y$"])

    for i in range(2):
        for j in range(2):
            val = cov_matrix_np[i, j]
            color = "white" if np.abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2e}", ha="center", va="center", color=color)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "fs_covariance_white_noise.pdf"
    plt.savefig(output_path)
    logger.info(f"Covariance matrix plot saved to '{output_path.resolve()}'")
    plt.close(fig)


if __name__ == "__main__":
    run_fs_covariance_analysis()
