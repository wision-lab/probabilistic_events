"""
Plotting helpers
"""

from typing import Dict

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import torch
from matplotlib import cm as cm

SAVEFIG_PDF_KWARGS = dict(dpi=150, transparent=True, bbox_inches="tight")

_colormap_lut_cache: Dict[str, torch.Tensor] = {}


def register_custom_cmaps():
    """
    Registers 3 discrete colormaps for sparse {-1, 0, 1} data.

    1. 'sparse_purple_green': High contrast, best for visibility.
    2. 'sparse_blue_orange':  Classic colorblind safe.
    3. 'sparse_pink_green':   Maximum brightness/neon look.
    """
    # Option A: Purple (#762a83) <-> White <-> Green (#1b7837)
    # Darker ends ensure they don't look washed out against white.
    c_pg = ["#762a83", "#ffffff", "#1b7837"]
    matplotlib.colormaps.register(
        name="sparse_purple_green", cmap=mcolors.ListedColormap(c_pg), force=True
    )

    # Option B: Blue (#0571b0) <-> White <-> Orange (#e66101)
    c_bo = ["#0571b0", "#ffffff", "#e66101"]
    matplotlib.colormaps.register(
        name="sparse_blue_orange", cmap=mcolors.ListedColormap(c_bo), force=True
    )

    # Option C: Pink (#e9a3c9) <-> White <-> YellowGreen (#a1d76a)
    # NOTE: These are lighter/brighter, but might be harder to see on white.
    c_piyg = [
        "#c51b7d",
        "#ffffff",
        "#4d9221",
    ]  # Using slightly darker versions for visibility
    matplotlib.colormaps.register(
        name="sparse_pink_green", cmap=mcolors.ListedColormap(c_piyg), force=True
    )


def apply_colormap_torch(
    tensor: torch.Tensor, colormap_name: str = "gray", num_colors: int = 256
) -> torch.Tensor:
    """
    Apply a matplotlib colormap to a torch tensor entirely on GPU.

    This function avoids GPU-CPU transfers by using cached GPU lookup tables
    for colormap application, providing access to matplotlib's full colormap
    collection with optimal performance.

    :param tensor:
        The input tensor of shape [H, W] with values normalized to [0, 1]
    :param colormap_name:
        Name of the matplotlib colormap. Supports all matplotlib colormaps including:
        - Perceptually uniform: viridis, plasma, inferno, magma, cividis
        - Sequential: Blues, Reds, Greens, etc.
        - Diverging: coolwarm, RdBu, RdYlBu, etc.
        - Cyclic: twilight, hsv
        - Qualitative: Set1, Set2, tab10, etc.

    :returns:
        RGB tensor of shape [H, W, 3] with colormap applied, values in [0, 255] uint8
    """
    device = tensor.device
    cache_key = f"{colormap_name}_{device}_{num_colors}"

    if cache_key not in _colormap_lut_cache:
        cmap = matplotlib.colormaps[colormap_name]
        colors = cmap(np.linspace(0, 1, num_colors))[:, :3]
        lut_tensor = torch.from_numpy(colors.astype(np.float32)).to(device)
        _colormap_lut_cache[cache_key] = lut_tensor

    colormap_lut = _colormap_lut_cache[cache_key]

    sanitized_tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

    # Convert tensor values to indices. Clamp is still good practice.
    indices = torch.clamp(sanitized_tensor * (num_colors - 1), 0, num_colors - 1).long()

    # Apply colormap
    colored_tensor = colormap_lut[indices]

    return (colored_tensor * 255).to(torch.uint8)




