import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from loguru import logger
from torch import Tensor

from probabilistic_events.ops.array_ops import create_1d_gaussian_kernels


class SeparableDifferenceOfGaussians(nn.Module):
    def __init__(
        self,
        num_scales: int = 4,
        base_sigma: float = 0.6,
        scale_multiplier: float = 2.0,
        elongation_ratio: float = 3.0,
        surround_ratio: float = 1.6,
        standardize: bool = True,
        max_kernel_radius: int = 128,
        padding_mode: Literal["reflect", "replicate", "constant"] = "replicate",
        subspace_components: Optional[int] = None,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.standardize = standardize
        self.padding_mode = padding_mode
        self.subspace_components = subspace_components

        self.register_buffer("is_fitted", torch.tensor(False, dtype=torch.bool))

        self.num_features = 2 * self.num_scales
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

        if self.subspace_components is not None:
            if self.subspace_components > self.num_features:
                self.subspace_components = self.num_features

        sigmas_center = []
        sigmas_surround = []
        sigmas_long = []
        normalization_factors = []

        for i in range(num_scales):
            sigma_center = base_sigma * (scale_multiplier**i)
            sigmas_center.append(sigma_center)
            sigmas_surround.append(sigma_center * surround_ratio)
            sigmas_long.append(sigma_center * elongation_ratio)
            normalization_factors.append(1.0 / sigma_center if standardize else 1.0)

        max_sigma = max(max(sigmas_surround), max(sigmas_long))
        radius = int(math.ceil(4.0 * max_sigma))
        if radius > max_kernel_radius:
            radius = max_kernel_radius

        self.kernel_size = 2 * radius + 1
        self.padding_amount = radius

        bandpass_list = []
        smooth_list = []

        for i in range(num_scales):
            t_dog = torch.tensor([sigmas_center[i], sigmas_surround[i]])
            k_dog, _ = create_1d_gaussian_kernels(
                self.kernel_size, t_dog, device=torch.device("cpu")
            )
            bandpass_list.append(k_dog[0] - k_dog[1])

            t_smooth = torch.tensor([sigmas_long[i]])
            k_smooth, _ = create_1d_gaussian_kernels(
                self.kernel_size, t_smooth, device=torch.device("cpu")
            )
            smooth_list.append(k_smooth.squeeze())

        self.raw_bandpass = nn.Parameter(
            torch.stack(bandpass_list), requires_grad=False
        )
        self.raw_smooth = nn.Parameter(torch.stack(smooth_list), requires_grad=False)
        self.scale_norms = nn.Parameter(
            torch.tensor(normalization_factors), requires_grad=False
        )

        vert_filters = []
        horiz_filters = []

        for i in range(num_scales):
            norm = normalization_factors[i]
            v1 = bandpass_list[i] * norm
            h1 = smooth_list[i]
            v2 = smooth_list[i] * norm
            h2 = bandpass_list[i]

            vert_filters.extend([v1, v2])
            horiz_filters.extend([h1, h2])

        self.fused_vert = nn.Parameter(
            torch.stack(vert_filters).view(2 * num_scales, 1, self.kernel_size, 1),
            requires_grad=False,
        )

        self.fused_horiz = nn.Parameter(
            torch.stack(horiz_filters).view(2 * num_scales, 1, 1, self.kernel_size),
            requires_grad=False,
        )

    @torch.jit.export
    def fit_subspace(self, regularization: float = 1e-4):
        device = self.fused_vert.device

        b_vecs = self.raw_bandpass.double() * self.scale_norms.view(-1, 1).double()
        s_vecs = self.raw_smooth.double() * self.scale_norms.view(-1, 1).double()

        dot_BB = b_vecs @ b_vecs.T
        dot_SS = s_vecs @ s_vecs.T
        dot_BS = b_vecs @ s_vecs.T
        dot_SB = dot_BS.T

        cov = torch.zeros(
            (self.num_features, self.num_features),
            dtype=torch.float64,
            device=device,
        )

        for i in range(self.num_scales):
            for j in range(self.num_scales):
                idx_v_i, idx_h_i = 2 * i, 2 * i + 1
                idx_v_j, idx_h_j = 2 * j, 2 * j + 1

                cov[idx_v_i, idx_v_j] = dot_BB[i, j] * dot_SS[i, j]
                cov[idx_h_i, idx_h_j] = dot_SS[i, j] * dot_BB[i, j]
                cov[idx_v_i, idx_h_j] = dot_BS[i, j] * dot_SB[i, j]
                cov[idx_h_i, idx_v_j] = dot_SB[i, j] * dot_BS[i, j]

        cov_reg = (
            cov
            + torch.eye(self.num_features, device=device, dtype=torch.float64)
            * regularization
        )
        vals, vecs = torch.linalg.eigh(cov_reg)

        sorted_idx = torch.argsort(vals, descending=True)
        sorted_vals = vals[sorted_idx].to(torch.float32)
        sorted_vecs = vecs[:, sorted_idx].to(torch.float32)

        eigenvalue_floor = sorted_vals[0].clamp_min(1e-6) * 1e-4
        safe_vals = sorted_vals.clamp_min(eigenvalue_floor)
        whitening_scale = torch.rsqrt(safe_vals)
        whitened_vecs = sorted_vecs * whitening_scale.unsqueeze(0)
        whitened_vecs = torch.nan_to_num(whitened_vecs, nan=0.0)

        self._cached_eigenvalues = sorted_vals
        self._cached_eigenvectors = whitened_vecs
        self.is_fitted.fill_(True)

    @torch.compile()
    def forward(
        self, image_input: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, "batch channels height width"]:
        pad = self.padding_amount
        x_padded = F.pad(image_input, (pad, pad, pad, pad), mode=self.padding_mode)
        feat_vert = F.conv2d(x_padded, self.fused_vert)
        return F.conv2d(feat_vert, self.fused_horiz, groups=self.num_features)

    @torch.compile()
    def forward_projected(
        self, image_input: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, "batch subspace_k height width"]:
        if not self.is_fitted:
            raise RuntimeError("Subspace not fitted. Call .fit_subspace() first.")

        features = self.forward(image_input)

        proj = self._cached_eigenvectors
        if self.subspace_components is not None:
            proj = proj[:, : self.subspace_components]

        weight = proj.T.view(proj.shape[1], proj.shape[0], 1, 1).to(features.dtype)

        return F.conv2d(features, weight)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Testing SeparableDifferenceOfGaussians on {device}")

    model = (
        SeparableDifferenceOfGaussians(num_scales=4, standardize=True).to(device).eval()
    )

    logger.info("Fitting subspace...")
    model.fit_subspace()

    dummy_input = torch.randn(1, 1, 256, 256, device=device)
    output = model.forward_projected(dummy_input)

    logger.info(f"Output shape: {output.shape}")
    logger.info("Variance of output channels (Target ~1.0):")
    logger.info(output.var(dim=(0, 2, 3)).detach().cpu().numpy())
