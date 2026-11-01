import numpy as np
import torch
from typing import Optional
from jaxtyping import Float
from scipy import spatial
from torch import Tensor
from torchvision.transforms.functional import gaussian_blur


def normalize_by_mean_and_quantile(
    image: Float[Tensor, "..."],
    reference: Float[Tensor, "..."] | None = None,
    quantile: float = 0.99,
    epsilon: float = 1e-8,
) -> Float[Tensor, "..."]:
    """Normalize an image by its mean and a robust white point.

    :param image: Image or image sequence to normalize.
    :param reference: Optional tensor whose mean sets the output mean.
    :param quantile: Quantile mapped to one.
    :param epsilon: Minimum safe denominator.
    :returns: Normalized image clamped to the unit interval.
    """
    normalized = image / image.mean().clamp_min(epsilon)
    if reference is not None:
        normalized = normalized * reference.mean().clamp_min(epsilon)
    flattened = normalized.flatten()
    if flattened.numel() > 1_000_000:
        sample_indices = torch.arange(
            0,
            flattened.numel(),
            step=max(1, flattened.numel() // 1_000_000),
            device=flattened.device,
        )[:1_000_000]
        flattened = flattened[sample_indices]
    white_point = torch.quantile(flattened, quantile).clamp_min(epsilon)
    return (normalized / white_point).clamp(0, 1)


def intensity_from_empirical_mean(
    empirical_mean: Float[Tensor, "..."],
    srgb_gamma_correct: bool = True,
    gamma_correction: float = 1.0,
    quantile: float = 0.99,
    return_linear: bool = False,
) -> Float[Tensor, "..."]:
    """Get intensities from empirical mean.

    Assumes zero-dark current. The output tensor will be on the same device
    as the input `empirical_mean` tensor.

    :param empirical_mean: Empirical mean across bit-planes (probability 0-1).
    :param srgb_gamma_correct: Whether to apply sRGB gamma correction.
    :param gamma_correction: Additional gamma correction factor.
    :param quantile: The quantile to map to 1.0 (Auto-Exposure).
                     Rec: 0.98-0.995 for HDR.
    :param return_linear: If True, returns raw HDR linear data (unclamped)
                          for tone mapping.
    :return: Processed intensity tensor on the same device as the input.
    """
    # 1. Safety Clamp: Prevent log(0) or log(negative)
    safe_mean = torch.clamp(empirical_mean, 0.0, 1.0 - 1e-6)

    # 2. Invert SPAD Response (Linearization)
    intensity = -torch.log(1.0 - safe_mean)

    # 3. Robust Auto-Exposure (Normalize by Quantile)
    if quantile is not None:
        white_point = torch.quantile(intensity, quantile)
        if white_point > 1e-8:
            intensity = intensity / white_point

    # 4. Return Selection
    if return_linear:
        # Return raw HDR data (values can be > 1.0)
        return intensity

    # --- LDR Fallback (if not tone mapping externally) ---
    intensity = torch.clamp(intensity, 0, 1)

    if srgb_gamma_correct:
        intensity = linear_to_srgb(intensity)

    if gamma_correction != 1.0:
        intensity = gamma_correct(intensity, gamma_correction)

    return intensity


def gamma_correct(
    img: Float[Tensor, "..."], gamma_correction: float = 1.0
) -> Float[Tensor, "..."]:
    """Apply gamma correction to an image tensor."""
    original_dtype = img.dtype

    if not torch.is_floating_point(img):
        img_float = img.float() / 255.0
        img_gamma = torch.pow(img_float, gamma_correction)
        img = img_gamma * 255.0
    else:
        img = torch.pow(img, gamma_correction)

    return img.to(original_dtype)


def srgb_to_linear(srgb: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Convert an sRGB tensor to a linear tensor."""
    original_dtype = srgb.dtype

    work_tensor = srgb
    if not srgb.is_floating_point():
        work_tensor = srgb.float() / 255.0

    linear = work_tensor.clone().clamp(0.0, 1.0)
    less = linear <= 0.04045

    linear[less] = linear[less] / 12.92
    linear[~less] = torch.pow((linear[~less] + 0.055) / 1.055, 2.4)

    if not original_dtype.is_floating_point:
        linear *= 255

    return linear.to(original_dtype)


def linear_to_srgb(linear: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Convert a linear tensor to an sRGB tensor."""
    original_dtype = linear.dtype

    work_tensor = linear
    if not linear.is_floating_point():
        work_tensor = linear.float() / 255.0

    srgb = work_tensor.clone()
    less = work_tensor <= 0.0031308

    srgb[less] = work_tensor[less] * 12.92
    srgb[~less] = 1.055 * torch.pow(work_tensor[~less], 1.0 / 2.4) - 0.055

    if not original_dtype.is_floating_point:
        srgb *= 255

    return srgb.to(original_dtype)


def nearest_neighbor_indices(hot_pixel_mask: np.ndarray):
    """Find source indices to inpaint from (one-time)."""
    i_ll, j_ll = np.where(1 - hot_pixel_mask)
    tree = spatial.KDTree(list(zip(i_ll.ravel(), j_ll.ravel())))
    query_i_ll, query_j_ll = np.where(hot_pixel_mask)
    query_ll = np.stack([query_i_ll, query_j_ll], axis=-1)

    dd, ii = tree.query(query_ll, workers=-1)
    nearest_ll = tree.data[ii].astype(int)
    nearest_i_ll, nearest_j_ll = nearest_ll[:, 0], nearest_ll[:, 1]
    return nearest_i_ll, nearest_j_ll


def nearest_neighbor_inpaint(array: torch.Tensor, hot_pixel_mask: np.ndarray):
    """Inpaint based on nearest neighbors."""
    i_ll, j_ll = np.where(1 - hot_pixel_mask)
    tree = spatial.KDTree(list(zip(i_ll.ravel(), j_ll.ravel())))
    query_i_ll, query_j_ll = np.where(hot_pixel_mask)
    query_ll = np.stack([query_i_ll, query_j_ll], axis=-1)

    dd, ii = tree.query(query_ll, workers=-1)
    nearest_ll = tree.data[ii].astype(int)
    if isinstance(array, torch.Tensor):
        nearest_ll = torch.from_numpy(nearest_ll).long().to(array.device)
    nearest_i_ll, nearest_j_ll = nearest_ll[:, 0], nearest_ll[:, 1]

    array[query_i_ll, query_j_ll] = array[nearest_i_ll, nearest_j_ll]
    return array


def ashikhmin_tonemap(
    image: torch.Tensor,
    epsilon: float = 1e-6,
    sigma: float = 10.0,
    kernel_size: int = 51,
) -> torch.Tensor:
    """
    Apply a simplified local tone mapping operator inspired by Ashikhmin (2002).
    Includes Per-Channel Auto-Exposure to fix green tint in SPAD data.
    """
    # 1. Sanitize input
    image = torch.nan_to_num(image, nan=0.0, posinf=None, neginf=0.0)
    image = torch.clamp(image, min=0.0)

    # 2. Reshape input to (1, C, H, W)
    if image.ndim == 2:
        img_input = image.unsqueeze(0).unsqueeze(0)
        is_color = False
    elif image.ndim == 3 and image.shape[2] == 3:
        img_input = image.permute(2, 0, 1).unsqueeze(0)
        is_color = True
    else:
        img_input = image.unsqueeze(0)
        is_color = True

    # 3. Extract Luminance
    if is_color and img_input.shape[1] == 3:
        R, G, B = img_input[:, 0:1], img_input[:, 1:2], img_input[:, 2:3]
        luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B
    else:
        luminance = img_input

    # 4. Calculate local adaptation (L_wa)
    if kernel_size % 2 == 0:
        kernel_size += 1

    log_luminance = torch.log(luminance + epsilon)
    log_blurred = gaussian_blur(
        log_luminance, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]
    )
    L_wa = torch.exp(log_blurred)

    # 5. Tone map
    tone_mapped = img_input / (1.0 + L_wa)

    # 6. Per-Channel Normalization (Fixes Green Tint)
    if is_color and tone_mapped.shape[1] == 3:
        # Flatten spatial dims: (1, 3, H*W)
        flat = tone_mapped.flatten(2)
        # Find 99th percentile PER CHANNEL: (1, 3, 1)
        white_point = torch.quantile(flat, 0.99, dim=2, keepdim=True).unsqueeze(-1)
        white_point = torch.clamp(white_point, min=epsilon)
        tone_mapped = tone_mapped / white_point
    else:
        # Global normalization for grayscale
        white_point = torch.quantile(tone_mapped, 0.99)
        if white_point > epsilon:
            tone_mapped = tone_mapped / white_point

    # 7. Final Clamp and Restore Shape
    tone_mapped = torch.clamp(tone_mapped, 0.0, 1.0)

    if is_color and image.ndim == 3 and image.shape[2] == 3:
        return tone_mapped.squeeze(0).permute(1, 2, 0)
    elif not is_color:
        return tone_mapped.squeeze(0).squeeze(0)

    return tone_mapped.squeeze(0)


def reinhard_2002_global_tonemap(
    image: torch.Tensor,
    intensity_key: float = 0.18,
    white_point: Optional[float] = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Apply Reinhard global tone mapping (2002).

    Calculates the log-average luminance and scales the image such that
    the geometric mean maps to `intensity_key`. It then compresses the
    dynamic range using the operator L / (1 + L).

    If `white_point` is provided, it uses the high-luminance operator:
    L * (1 + L / L_white^2) / (1 + L), which allows burning out highlights
    (mapping L_white to infinity -> 1.0 after clamping).

    Args:
        image: Linear RGB or Grayscale tensor.
               Shapes: (H, W), (H, W, 3), (C, H, W), or (B, C, H, W).
        intensity_key: "a" parameter. Subjective mid-gray intensity (default 0.18).
        white_point: The smallest luminance value that will be mapped to pure white.
                     If None, the simple operator L / (1+L) is used.
        epsilon: Small constant to avoid log(0).

    Returns:
        Tone mapped image in [0, 1] range. Same shape as input.
    """
    # 1. Sanitize input
    image = torch.nan_to_num(image, nan=0.0, posinf=None, neginf=0.0)
    image = torch.clamp(image, min=0.0)

    # 2. Reshape input to (B, C, H, W)
    if image.ndim == 2:  # (H, W)
        x = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        if image.shape[2] == 3:  # (H, W, 3) -> (1, 3, H, W)
            x = image.permute(2, 0, 1).unsqueeze(0)
        else:  # (C, H, W) -> (1, C, H, W)
            x = image.unsqueeze(0)
    else:  # (B, C, H, W)
        x = image

    # 3. Calculate Luminance (L_w)
    # If 3 channels, use Rec. 709 coeffs. Else assume already luminance.
    if x.shape[1] == 3:
        L_w = (
            0.2126 * x[:, 0:1, :, :]
            + 0.7152 * x[:, 1:2, :, :]
            + 0.0722 * x[:, 2:3, :, :]
        )
    else:
        L_w = x.mean(dim=1, keepdim=True)

    # 4. Log-Average Luminance (L_avg) per image in batch
    # Geometric mean = exp( arithmetic_mean( log(L + eps) ) )
    # We compute mean over spatial dims (H, W)
    log_L = torch.log(L_w + epsilon)
    L_avg = torch.exp(torch.mean(log_L, dim=(2, 3), keepdim=True))

    # 5. Scaled Luminance (L(x,y))
    # Equation 2: L(x,y) = (a / L_avg) * L_w(x,y)
    L_scaled = (intensity_key / (L_avg + epsilon)) * L_w

    # 6. Tone Mapping Operator
    if white_point is not None:
        # Equation 4: High luminance burnout
        # L_d = (L * (1 + L / L_white^2)) / (1 + L)
        L_d = (L_scaled * (1 + L_scaled / (white_point**2))) / (1 + L_scaled)
    else:
        # Equation 3: Simple operator
        # L_d = L / (1 + L)
        L_d = L_scaled / (1 + L_scaled)

    # 7. Reconstruct Color (L_d applies to Luminance, preserve ratios)
    # C_out = C_in * (L_d / L_w)
    # Optimization: L_d / L_w = scale_factor / (1 + ...)
    scale = L_d / (L_w + epsilon)

    # Broadcast scale (B, 1, H, W) to (B, C, H, W)
    output = x * scale

    # 8. Clamp and Restore Shape
    output = torch.clamp(output, 0.0, 1.0)

    if image.ndim == 2:
        return output.squeeze()
    elif image.ndim == 3 and image.shape[2] == 3:
        return output.squeeze(0).permute(1, 2, 0)
    elif image.ndim == 3:
        return output.squeeze(0)

    return output


_PU21_A = 0.001908
_PU21_B = 0.0078
_PU21_L_MIN = 0.005
_PU21_L_MAX = 10000.0
_PU21_L_MIN_LOG2 = torch.log2(torch.tensor(_PU21_L_MIN))


def pu21_encode(luminance: Tensor) -> Tensor:
    """Encode absolute luminance with the PU21 display transfer."""
    luminance_clamped = torch.clip(luminance, _PU21_L_MIN, _PU21_L_MAX)
    log_domain = torch.log2(luminance_clamped) - _PU21_L_MIN_LOG2.to(
        device=luminance.device,
        dtype=luminance.dtype,
    )
    return _PU21_A * log_domain.square() + _PU21_B * log_domain


def compute_luminance(rgb: Tensor) -> Tensor:
    """Compute Rec.709 luminance for an ``[..., 3]`` RGB tensor."""
    rgb = rgb.float()
    return rgb[..., 0:1] * 0.2126 + rgb[..., 1:2] * 0.7152 + rgb[..., 2:3] * 0.0722


def pu21_encode_rgb_luminance_corrected(rgb_abs: Tensor) -> Tensor:
    """Apply PU21 to luminance while preserving RGB chromatic ratios."""
    rgb_abs = rgb_abs.float()
    if rgb_abs.numel() == 0:
        return torch.full_like(rgb_abs, 0.5, dtype=torch.float32)

    luminance = compute_luminance(rgb_abs)
    encoded_luminance = pu21_encode(luminance)
    color_ratios = rgb_abs / luminance.clamp(min=1.0e-6)
    return encoded_luminance * color_ratios


def scale_to_L_peak(rgb: Tensor, L_peak: float = 4000.0) -> tuple[Tensor, float]:
    """Scale a linear image so its brightest value reaches ``L_peak`` nits."""
    rgb_safe = torch.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)

    if rgb_safe.ndim == 4 and rgb_safe.shape[1] in {1, 3}:
        rgb_bhwc = rgb_safe.permute(0, 2, 3, 1)
        output_layout = "bchw"
    elif rgb_safe.ndim == 4 and rgb_safe.shape[-1] in {1, 3}:
        rgb_bhwc = rgb_safe
        output_layout = "bhwc"
    elif rgb_safe.ndim == 3 and rgb_safe.shape[0] in {1, 3}:
        rgb_bhwc = rgb_safe.permute(1, 2, 0).unsqueeze(0)
        output_layout = "chw"
    elif rgb_safe.ndim == 3 and rgb_safe.shape[-1] in {1, 3}:
        rgb_bhwc = rgb_safe.unsqueeze(0)
        output_layout = "hwc"
    elif rgb_safe.ndim == 2:
        rgb_bhwc = rgb_safe.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        output_layout = "hw"
    else:
        raise ValueError(f"Unexpected input shape: {tuple(rgb.shape)}")

    max_val = float(rgb_bhwc.max().item()) if rgb_bhwc.numel() > 0 else 0.0
    if max_val <= 0.0:
        scaled = torch.full_like(rgb_bhwc, _PU21_L_MIN)
        scale_factor = 1.0
    else:
        scale_factor = float(L_peak) / max_val
        scaled = rgb_bhwc * scale_factor

    if output_layout == "bchw":
        return scaled.permute(0, 3, 1, 2), scale_factor
    if output_layout == "bhwc":
        return scaled, scale_factor
    if output_layout == "chw":
        return scaled.squeeze(0).permute(2, 0, 1), scale_factor
    if output_layout == "hwc":
        return scaled.squeeze(0), scale_factor
    return scaled.squeeze(0)[..., 0], scale_factor
