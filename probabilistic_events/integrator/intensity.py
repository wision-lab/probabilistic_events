from math import pi, log, floor
from typing import Literal

import torch
from einops import rearrange, reduce
from jaxtyping import Float, Int
from loguru import logger
from torch import Tensor
from torch import autocast

from probabilistic_events.denoiser.adaptive_wiener import AdaptiveWienerFilter
from probabilistic_events.filters.difference_of_gaussians import (
    SeparableDifferenceOfGaussians,
)
from probabilistic_events.filters.farid_and_simoncelli import FaridSimoncelli
from probabilistic_events.filters.log_gabor import LogGabor
from probabilistic_events.filters.malvar_he_cutler import (
    MalvarHeCutlerDemosaic,
    BayerPatternLiteral,
)
from probabilistic_events.integrator.base import (
    BaseMotionMetering,
    update_tensor,
    log_beta,
    log_binom,
)
from probabilistic_events.integrator.triton_kernels import (
    HAS_TRITON,
    bocpd_binomial_triton,
)
from probabilistic_events.ops.array_ops import binomial_vst
from probabilistic_events.ops.color import GrayWorldWB
from probabilistic_events.ops.image import (
    intensity_from_empirical_mean,
    ashikhmin_tonemap as ashikhmin_tonemap_fn,
    reinhard_2002_global_tonemap as reinhard_tonemap_fn,
    linear_to_srgb,
    pu21_encode,
    pu21_encode_rgb_luminance_corrected,
    scale_to_L_peak,
)

if hasattr(torch, "_dynamo") and hasattr(
    torch._dynamo.config,
    "allow_unspec_int_on_nn_module",
):
    torch._dynamo.config.allow_unspec_int_on_nn_module = True


class ProbabilisticEvents(BaseMotionMetering):
    """
    Bayesian Online Changepoint Detection (BOCPD) for intensity estimation.

    This class implements a metered reconstruction of image intensity by modeling
    temporal evolution with a Run-Length distribution. It supports various feature
    extractors (Gradient, LogGabor, DoG) to guide the changepoint detection.
    """

    def __init__(
        self,
        bayer_pattern: BayerPatternLiteral = "gray",
        bocpd_gamma: float = 1e-5,
        sum_frames: int = 31,
        memory_size: int = 5,
        model_binomial: bool = True,
        model_gradient: bool = False,
        model_log_gabor: bool = True,
        model_difference_of_gaussians: bool = False,
        max_filter_size: int = 1,
        num_scales: int = 4,
        num_orientations: int = 8,
        pca_components: int | None = 24,
        denoise: bool = False,
        denoise_window_size: int = 7,
        denoise_noise_gain: float = 0.8,
        white_balance: bool = True,
        linearize_response: bool = False,
        flux_est_quantile: float = 0.98,
        use_bayesian_predictive: bool = False,
        tonemap: Literal[
            "srgb",
            "ashikhmin",
            "reinhard_2002_global",
            "pu21_luminance",
            "",
        ] = "",
        entropy_event_threshold: float = 1.4,
        backend: Literal["auto", "torch", "triton"] = "auto",
    ):
        """
        Initialize the ProbabilisticEvents estimator.

        Defaults match ``conf/method.yaml`` and the paper's Methods section
        (gamma = 1e-5, K = 5, binomial frames of N_sum = 31, Log-Gabor features).
        """
        super().__init__(
            bocpd_gamma=bocpd_gamma,
            sum_frames=sum_frames,
            memory_size=memory_size,
            min_filter_size=max_filter_size,
        )

        self.max_filter_size = max_filter_size
        self.gradient_kernel_size = 5

        if model_gradient:
            logger.warning("Overriding num_scales to be 1")
            num_scales = 1

        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.pca_components = pca_components

        self.alpha_0 = 0.5
        self.beta_0 = 0.5

        self.linearize_response = linearize_response
        self.flux_est_quantile = flux_est_quantile

        # Validate and set tonemapping mode
        valid_tonemaps = {
            "srgb",
            "ashikhmin",
            "reinhard_2002_global",
            "pu21_luminance",
            "none",
        }
        self.tonemap = str(tonemap).lower() if tonemap in valid_tonemaps else None

        if self.tonemap == "none":
            self.tonemap = None

        self.use_bayesian_predictive = use_bayesian_predictive

        # The threshold is a factor on the run-length perplexity exp(H), so the
        # entropy difference is compared against log(factor) nats (2.0 -> 0.69)
        self.entropy_event_threshold = log(max(1.001, entropy_event_threshold))
        self.backend = backend

        self.model_binomial = model_binomial
        self.collect_binomial_stats = model_binomial or use_bayesian_predictive
        self.model_gradient = model_gradient
        self.model_log_gabor = model_log_gabor
        self.model_difference_of_gaussians = model_difference_of_gaussians

        num_feature_models = (
            int(model_gradient)
            + int(model_log_gabor)
            + int(model_difference_of_gaussians)
        )
        if num_feature_models == 0:
            logger.debug("No feature models active.")
        if num_feature_models > 1:
            raise ValueError("Only one feature model can be active.")

        self.bayer_pattern = bayer_pattern.lower()
        self.is_bayer = self.bayer_pattern not in ["gray", "grey"]

        self.demosaic_std_module = MalvarHeCutlerDemosaic(
            bayer_pattern=self.bayer_pattern
        )
        self.post_demosaic_fn = self.demosaic_std_module.demosaic_linear

        self.white_balance = None
        if self.is_bayer and white_balance:
            self.white_balance = GrayWorldWB()

        self.gradient_filter = FaridSimoncelli(
            num_tap=self.gradient_kernel_size, num_order=1, standardize=True
        )
        self.gradient_filter.eval().requires_grad_(False)

        self.log_gabor_filter = None
        if self.model_log_gabor:
            self.log_gabor_filter = LogGabor(
                num_scales=num_scales,
                num_orientations=num_orientations,
                standardize=True,
                fft_artifact_handling="padding",
                subspace_components=self.pca_components,
            )
            self.log_gabor_filter.eval()

        self.difference_of_gaussians_filter = None
        if self.model_difference_of_gaussians:
            self.difference_of_gaussians_filter = SeparableDifferenceOfGaussians(
                num_scales=num_scales,
                standardize=True,
                subspace_components=self.pca_components,
            )
            self.difference_of_gaussians_filter.eval()

        self.active_feature_filter = None
        if self.model_gradient:
            self.active_feature_filter = self.gradient_filter
        elif self.model_log_gabor:
            self.active_feature_filter = self.log_gabor_filter
        elif self.model_difference_of_gaussians:
            self.active_feature_filter = self.difference_of_gaussians_filter

        self.denoise = denoise
        self.wiener_filter = None
        if self.denoise:
            self.wiener_filter = AdaptiveWienerFilter(
                window_size=denoise_window_size, noise_gain=denoise_noise_gain
            )
            self.wiener_filter.eval()

        self.insert_index: Int[Tensor, "h w"] | None = None
        self.insert_mask: Tensor | None = None
        self.alpha_ll: Float[Tensor, "h w m"] | None = None
        self.beta_ll: Float[Tensor, "h w m"] | None = None
        self.samples_ll: Float[Tensor, "h w m"] | None = None
        self.feature_mu_ll: Float[Tensor, "c h w m"] | None = None
        self.prev_forecaster_entropy: Float[Tensor, "h w"] | None = None

        if self.max_filter_size > 1:
            self._pool_fn = self.max_filter_2d
        else:
            self._pool_fn = lambda x: x

    def __repr__(self):
        models = []
        if self.model_binomial:
            models.append("binomial")
        elif self.collect_binomial_stats:
            models.append("binomial stats")

        if self.model_gradient:
            models.append("gradient")
        if self.model_log_gabor:
            k_str = f"k={self.pca_components}" if self.pca_components else "full"
            models.append(f"log_gabor ({k_str})")
        if self.model_difference_of_gaussians:
            k_str = f"k={self.pca_components}" if self.pca_components else "full"
            models.append(f"difference_of_gaussians ({k_str})")

        mode = "Bayesian Predictive" if self.use_bayesian_predictive else "Adaptive EMA"

        return (
            f"{self.__class__.__name__}(gamma={self.bocpd_gamma}, "
            f"memory={self.memory_size}, pattern={self.bayer_pattern}, "
            f"models=[{', '.join(models)}], reconstruction={mode}, denoise={self.denoise}, tonemap={self.tonemap})"
        )

    @property
    def total_occupied(self) -> int:
        """Get the number of currently active run-length hypotheses."""
        return min(floor(self.t_absolute / self.sum_frames) + 1, self.memory_size)

    def reset(self):
        """Reset the internal state of the estimator."""
        super().reset()
        self.feature_mu_ll = None
        self.prev_forecaster_entropy = None

    def _initialize_state_tensors(self, photon_cube: Float[Tensor, "h w t"]):
        """Initialize the tensors required for BOCPD tracking."""
        spatial_dims = photon_cube.shape[:-1]
        bocpd_shape = (*spatial_dims, self.memory_size)
        device = photon_cube.device
        dtype = torch.float32

        self.log_forecaster_distribution_ll = torch.full(
            bocpd_shape, -1000.0, device=device, dtype=dtype
        )
        self.log_forecaster_distribution_ll[..., 0] = 0.0

        self.forecaster_distribution_index_ll = torch.full(
            bocpd_shape, 0, device=device, dtype=dtype
        )
        self.insert_index = torch.full(
            spatial_dims, -1, device=device, dtype=torch.long
        )
        self.insert_mask = torch.zeros(spatial_dims, device=device, dtype=torch.bool)
        self.recons = torch.zeros(spatial_dims, device=device, dtype=dtype)
        self.alpha_ll = torch.full(
            bocpd_shape, self.alpha_0, device=device, dtype=dtype
        )
        self.beta_ll = torch.full(bocpd_shape, self.beta_0, device=device, dtype=dtype)
        self.samples_ll = torch.ones(bocpd_shape, device=device, dtype=dtype)
        self.feature_mu_ll = None

        self.prev_forecaster_entropy = None

        self.gradient_filter.to(device)
        for filt in [
            self.log_gabor_filter,
            self.difference_of_gaussians_filter,
        ]:
            if filt:
                filt.to(device)
                if hasattr(filt, "fit_subspace"):
                    filt.fit_subspace()

        if self.wiener_filter:
            self.wiener_filter.to(device)
        if self.demosaic_std_module is not None:
            self.demosaic_std_module.to(device).eval()
        if self.white_balance is not None:
            self.white_balance.to(device).eval()

    @torch.compile()
    def _extract_features(
        self, photon_cube_vst: Float[Tensor, "t 1 h w"]
    ) -> Float[Tensor, "c h w t"] | None:
        """Extract features from variance-stabilized input."""
        if self.model_gradient:
            grad_x_cube, grad_y_cube = self.gradient_filter(photon_cube_vst)
            features = torch.stack(
                [grad_x_cube.squeeze(1), grad_y_cube.squeeze(1)], dim=1
            )
        elif self.active_feature_filter is not None:
            if hasattr(self.active_feature_filter, "forward_projected"):
                features = self.active_feature_filter.forward_projected(photon_cube_vst)
            else:
                features = self.active_feature_filter(photon_cube_vst)
        else:
            return None
        return rearrange(features, "time k height width -> k height width time")

    def _compute_forecaster_entropy(self) -> Float[Tensor, "h w"]:
        """Compute the entropy of the run-length distribution."""
        occupied = self.total_occupied
        hypothesis_indices = torch.arange(
            self.memory_size,
            device=self.log_forecaster_distribution_ll.device,
        )
        active_mask = hypothesis_indices < occupied
        active_logits = self.log_forecaster_distribution_ll.masked_fill(
            ~active_mask,
            float("-inf"),
        )
        probs = torch.softmax(active_logits, dim=-1).masked_fill(~active_mask, 0.0)
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

    def _compute_bayesian_mean(self) -> Float[Tensor, "h w"]:
        """Compute the expected value of the marginal predictive distribution."""
        if not self.collect_binomial_stats:
            return self.recons

        # P(r_t | x_{1:t}) - The Run Length Posterior
        run_length_probs = torch.softmax(self.log_forecaster_distribution_ll, dim=-1)
        total_counts = self.alpha_ll + self.beta_ll
        predictive_mean_r = self.alpha_ll / (total_counts + 1e-10)

        # Marginalize
        bayesian_mean = torch.sum(predictive_mean_r * run_length_probs, dim=-1)

        return bayesian_mean

    @torch.no_grad()
    def forward(self, photon_cube: Float[Tensor, "h w t"]) -> tuple[
        Float[Tensor, "h w c"] | Float[Tensor, "h w"],
        Float[Tensor, "h w"],
        Float[Tensor, "h w"],
        Int[Tensor, "h w"],
    ]:
        """Process a batch of photon frames."""
        device = photon_cube.device
        with autocast(device_type=device.type):
            if self.t_absolute == 0:
                self.reset()
                self._initialize_state_tensors(photon_cube)

            features_cube = None
            if self.active_feature_filter is not None:
                vst_cube = binomial_vst(photon_cube.float(), self.sum_frames)
                vst_bchw = rearrange(vst_cube, "height width time -> time 1 height width")
                if self.is_bayer:
                    vst_bchw = self.demosaic_std_module(vst_bchw)
                features_cube = self._extract_features(vst_bchw)

            if features_cube is not None and self.feature_mu_ll is None:
                c = features_cube.shape[0]
                spatial = photon_cube.shape[:-1]
                self.feature_mu_ll = torch.zeros(
                    (c, *spatial, self.memory_size), device=device
                )

            # Core BOCPD Loop
            if self._use_triton(photon_cube, features_cube):
                raw_recons, sample_weight, entropy_vis = self._core_loop_triton(
                    photon_cube, features_cube
                )
            else:
                raw_recons, sample_weight, entropy_vis = self._core_loop(
                    photon_cube, features_cube
                )

            # Apply ISP Pipeline
            # Returns: H W C (if Color) or H W (if Mono)
            recons = self.apply_isp(raw_recons, sample_weight)

            # Calculate gradients on the monochrome mean of the final image for viz
            # We standarize to (1, 1, H, W) for the filter
            if recons.ndim == 3:
                # H W C -> 1 1 H W
                recons_mono = reduce(recons, "h w c -> 1 1 h w", "mean")
            else:
                # H W -> 1 1 H W
                recons_mono = rearrange(recons, "h w -> 1 1 h w")

            grad_x, grad_y = self.gradient_filter(recons_mono)
            grad_abs = torch.hypot(grad_x, grad_y).squeeze()
            grad_abs = grad_abs.clamp(0, 2) / 2

            return recons, grad_abs, sample_weight, entropy_vis

    def apply_isp(
        self,
        recons: Float[Tensor, "h w"],
        sample_weight: Float[Tensor, "h w"] | None = None,
    ) -> Float[Tensor, "h w c"] | Float[Tensor, "h w"]:
        """
        Apply the ISP pipeline to the reconstructed raw image.

        Pipeline Order:
        1. Demosaic (if applicable) -> Moves data to Spatial RGB.
        2. Denoise (Wiener Filter) -> Operates on Probability/Count space (p ~ 0-1).
        3. Linearize Response -> Converts Probability to Linear Flux/Intensity.
        4. White Balance -> Adjusts linear flux channels.
        5. Tonemap -> Compresses range for display.
        """
        # 1. Expand to BCHW: (H, W) -> (1, 1, H, W)
        # We keep (1, C, H, W) throughout the pipeline for consistency.
        out = rearrange(recons, "h w -> 1 1 h w")

        # 2. Demosaic
        if self.is_bayer:
            # Output becomes (1, 3, H, W)
            out = self.post_demosaic_fn(out)

        with torch.autocast(device_type=recons.device.type, enabled=False):
            # 3. Denoise
            # Operates in Probability/Count space (before linearization)
            if (
                self.denoise
                and self.wiener_filter is not None
                and sample_weight is not None
            ):
                # Calculate effective accumulated sample count (n)
                p_fail = sample_weight.clamp(min=self._eps, max=1.0 - 1e-7)
                run_length = -1.0 / torch.log(1.0 - p_fail)
                n_est = run_length * self.sum_frames

                # AdaptiveWienerFilter accepts (H, W) or (C, H, W), not BCHW.
                out = self.wiener_filter(out.squeeze(0), n_est).unsqueeze(0)

            # 4. Linearize Response
            if self.linearize_response:
                out = intensity_from_empirical_mean(
                    out,
                    quantile=self.flux_est_quantile,
                    return_linear=True,
                )

            # 5. White Balance
            if self.is_bayer and self.white_balance is not None:
                out = self.white_balance(out)

            # 6. Tonemap
            if self.tonemap == "ashikhmin":
                out = ashikhmin_tonemap_fn(out)
            elif self.tonemap == "reinhard_2002_global":
                out = reinhard_tonemap_fn(out)
            elif self.tonemap == "srgb":
                out = linear_to_srgb(out)
            elif self.tonemap == "pu21_luminance":
                out_nit_scaled, _ = scale_to_L_peak(out)
                if out_nit_scaled.shape[1] == 1:
                    out = pu21_encode(out_nit_scaled)
                else:
                    out = pu21_encode_rgb_luminance_corrected(out_nit_scaled)

        # 7. Final Reshape to H W C or H W
        if self.is_bayer:
            # Color: 1 3 H W -> H W 3
            return rearrange(out, "1 c h w -> h w c")
        else:
            # Mono: 1 1 H W -> H W
            return rearrange(out, "1 1 h w -> h w")

    @torch.compile()
    def _core_loop(
        self,
        photon_cube: Float[Tensor, "h w t"],
        features_cube: Float[Tensor, "c h w t"] | None,
    ) -> tuple[Float[Tensor, "h w"], Float[Tensor, "h w"], Int[Tensor, "h w"]]:
        """Execute the main time-step loop for BOCPD."""
        sample_weight = torch.ones_like(self.recons)

        n_accumulated = self.sum_frames
        num_steps = photon_cube.shape[-1]

        for t in range(num_steps):
            photon_slice = photon_cube[..., t].float()
            self.t_absolute += self.sum_frames

            sample_weight = self.estimate_sample_weight()
            sample_weight = self._pool_fn(sample_weight)

            if not self.use_bayesian_predictive:
                self._reconstruct_adaptive_ema(sample_weight, photon_slice)

            feat_slice = features_cube[..., t] if features_cube is not None else None
            log_likelihood = self.get_posterior_predictive(
                photon_slice, feat_slice, n_accumulated
            )
            self.update_forecasters(log_likelihood)
            self.update_prior(photon_slice, feat_slice, n_accumulated)

        entropy_vis = self._entropy_events()
        return self.recons, sample_weight, entropy_vis

    def _use_triton(self, photon_cube, features_cube) -> bool:
        """Whether the fused Triton kernel supports the current configuration."""
        if self.backend == "torch":
            return False
        supported = (
            HAS_TRITON
            and photon_cube.is_cuda
            and (self.model_binomial or features_cube is not None)
            and self.max_filter_size == 1
        )
        if self.backend == "triton" and not supported:
            raise ValueError(
                "backend='triton' needs CUDA, a binomial or feature likelihood "
                "and max_filter_size=1"
            )
        return supported

    def _core_loop_triton(
        self,
        photon_cube: Float[Tensor, "h w t"],
        features_cube: Float[Tensor, "c h w t"] | None,
    ):
        """Fused-kernel equivalent of `_core_loop`."""
        has_prev = self.prev_forecaster_entropy is not None
        if not has_prev:
            self.prev_forecaster_entropy = torch.empty_like(self.recons)
        (
            sample_weight,
            self.insert_index,
            self.insert_mask,
            entropy_vis,
        ) = bocpd_binomial_triton(
            photon_cube,
            self.log_forecaster_distribution_ll,
            self.forecaster_distribution_index_ll,
            self.alpha_ll,
            self.beta_ll,
            self.recons,
            t_abs0=self.t_absolute,
            gamma=self.bocpd_gamma,
            sum_frames=self.sum_frames,
            alpha_0=self.alpha_0,
            beta_0=self.beta_0,
            use_ema=not self.use_bayesian_predictive,
            entropy=self.prev_forecaster_entropy,
            has_prev_entropy=has_prev,
            event_threshold=self.entropy_event_threshold,
            binomial=self.model_binomial,
            collect=self.collect_binomial_stats,
            features=features_cube,
            mu=self.feature_mu_ll,
            samples=self.samples_ll,
        )
        self.t_absolute += self.sum_frames * photon_cube.shape[-1]
        return self.recons, sample_weight, entropy_vis

    def _entropy_events(self) -> Int[Tensor, "h w"]:
        """Entropy-change event map, plus the Bayesian-predictive reconstruction."""
        current_entropy = self._compute_forecaster_entropy()

        # Categorical Event Map: 0 = Static, 1 = Surprise, -1 = Resolution
        entropy_vis = torch.zeros_like(self.recons, dtype=torch.long)

        if self.prev_forecaster_entropy is not None:
            delta = current_entropy - self.prev_forecaster_entropy
            entropy_vis[delta > self.entropy_event_threshold] = 1
            entropy_vis[delta < -self.entropy_event_threshold] = -1

        self.prev_forecaster_entropy = current_entropy.detach()

        if self.use_bayesian_predictive and self.collect_binomial_stats:
            self.recons = self._compute_bayesian_mean()

        return entropy_vis

    @torch.compile()
    def get_posterior_predictive(self, photon_slice, features_slice, n_accumulated):
        log_likelihood = torch.zeros_like(self.alpha_ll)

        if self.model_binomial:
            n_trials = torch.tensor(
                n_accumulated, dtype=torch.float32, device=photon_slice.device
            )

            log_likelihood += (
                log_binom(n_trials, photon_slice).unsqueeze(-1)
                - log_beta(self.alpha_ll, self.beta_ll)
                + log_beta(
                    self.alpha_ll + photon_slice.unsqueeze(-1),
                    self.beta_ll + n_trials - photon_slice.unsqueeze(-1),
                )
            )

        if features_slice is None or self.feature_mu_ll is None:
            return log_likelihood

        num_channels = self.feature_mu_ll.shape[0]

        predictive_variance = (self.samples_ll + 1) / self.samples_ll
        squared_error = (features_slice.unsqueeze(-1) - self.feature_mu_ll).pow(2)
        mahalanobis_distance = squared_error.sum(dim=0) / predictive_variance

        log_determinant = num_channels * torch.log(predictive_variance)
        log_2pi_constant = num_channels * log(2 * pi)
        log_likelihood -= 0.5 * (
            log_2pi_constant + log_determinant + mahalanobis_distance
        )

        return log_likelihood

    def update_prior(
        self,
        photon_slice: Float[Tensor, "h w"],
        features_slice: Float[Tensor, "c h w"] | None,
        n_accumulated: float,
    ):
        """Update the sufficient statistics of the prior distributions."""
        # Only reset slots where the new r=0 hypothesis was actually inserted
        mask = self.insert_mask

        if self.collect_binomial_stats:
            update_tensor(self.alpha_ll, self.insert_index, self.alpha_0, mask)
            update_tensor(self.beta_ll, self.insert_index, self.beta_0, mask)

        if features_slice is not None and self.feature_mu_ll is not None:
            update_tensor(self.samples_ll, self.insert_index, 0.0, mask)
            # Expand index for channel dimension
            idx = rearrange(self.insert_index, "h w -> 1 h w")
            mask_expanded = rearrange(mask, "h w -> 1 h w")
            update_tensor(self.feature_mu_ll, idx, 0.0, mask_expanded)

        if self.collect_binomial_stats:
            ph_expanded = rearrange(photon_slice, "h w -> h w 1")
            self.alpha_ll += ph_expanded
            self.beta_ll += (n_accumulated - photon_slice).unsqueeze(-1)

        if features_slice is not None and self.feature_mu_ll is not None:
            feat_expanded = rearrange(features_slice, "c h w -> c h w 1")
            samples_expanded = rearrange(self.samples_ll, "h w k -> 1 h w k")

            self.samples_ll += 1
            self.feature_mu_ll += (
                feat_expanded - self.feature_mu_ll
            ) / samples_expanded

    def update_forecasters(self, log_likelihood_ll: Float[Tensor, "h w m"]):
        """Update the run-length probabilities and manage hypothesis pruning."""
        occupied = self.total_occupied
        hypothesis_indices = torch.arange(
            self.memory_size,
            device=self.log_forecaster_distribution_ll.device,
        )
        active_mask = hypothesis_indices < occupied
        inactive_mask = ~active_mask
        active_ll = self.log_forecaster_distribution_ll.masked_fill(
            inactive_mask,
            float("-inf"),
        )
        active_log_lik = log_likelihood_ll.masked_fill(inactive_mask, float("-inf"))

        new_forecaster = log(self.bocpd_gamma) + torch.logsumexp(
            active_ll + active_log_lik, dim=-1
        )
        continued_forecasters = self.log_forecaster_distribution_ll + (
            log(1 - self.bocpd_gamma) + active_log_lik
        )
        self.log_forecaster_distribution_ll = torch.where(
            active_mask,
            continued_forecasters,
            self.log_forecaster_distribution_ll,
        )

        _, self.insert_index = torch.min(self.log_forecaster_distribution_ll, dim=-1)

        # Select prob to beat using gather
        idx_expanded = rearrange(self.insert_index, "h w -> h w 1")
        prob_to_beat = torch.gather(
            self.log_forecaster_distribution_ll, -1, idx_expanded
        ).squeeze(-1)

        mask = new_forecaster >= prob_to_beat
        self.insert_mask = mask

        update_tensor(
            self.log_forecaster_distribution_ll, self.insert_index, new_forecaster, mask
        )
        update_tensor(
            self.forecaster_distribution_index_ll,
            self.insert_index,
            self.t_absolute,
            mask,
        )

        # Normalize log probs to prevent numerical instability
        max_vals = reduce(self.log_forecaster_distribution_ll, "h w k -> h w 1", "max")
        self.log_forecaster_distribution_ll -= max_vals

    def _reconstruct_adaptive_ema(
        self, sample_weight: Float[Tensor, "h w"], photon_slice: Float[Tensor, "h w"]
    ):
        """Update the reconstruction using Adaptive Exponential Moving Average."""
        new_data = photon_slice / self.sum_frames
        self.recons = (1 - sample_weight) * self.recons + sample_weight * new_data
