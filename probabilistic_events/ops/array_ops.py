"""
Array ops
"""

from math import ceil, floor, sqrt
from math import log2
from random import random
from typing import Tuple

import numpy as np
import torch
from numba import vectorize, int64, float64
from torch import Tensor


def normalize_uint8(img):
    return float_to_uint8(normalize(img))


def normalize(img):
    if img.max() == img.min():
        return torch.ones_like(img) * 0.5
    return (img - img.min()) / (img.max() - img.min())


def float_to_uint8(img):
    if np.issubdtype(img.dtype, np.floating):
        return (img * 255).astype(np.uint8)
    else:
        return img.astype(np.uint8)


@vectorize([int64(float64)])
def dither_round(x: float):
    """
    Rounding strategy based on dithering

    Ensures E[dither_round(x)] = x
    i.e., unbiased.

    :param x: floating point value
    :return: floor(x) + e
        e ~ Ber(fractional part of x)
    """
    x_int = floor(x)
    x_frac = x - x_int
    e = random() < x_frac
    return x_int + e


def nearest_power_of_two(number):
    # Returns next power of two following 'number'
    return pow(2, ceil(log2(number)))


def zero_pad_to_power_2(array, axis: Tuple = None):
    """
    Pads array to ensure dims specifed by axis are powers of 2

    Pads at the end, with zeros

    :param array: np.ndarray
    :param axis: int or sequence of ints
    :return:
    """
    if isinstance(axis, int):
        axis = (axis,)

    pad_ll = []
    for e, dim in enumerate(array.shape):
        if e in axis:
            dim_deficit = nearest_power_of_two(dim) - dim
            pad_ll.append([0, dim_deficit])
        else:
            pad_ll.append([0, 0])

    return np.pad(array, pad_ll)


def torch_quantile(
    input: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)


def anscombe_poisson_vst(x):
    return 2 * torch.sqrt(x + 3 / 8)


def anscombe_poisson_inverseVST(x):
    """
    Inverse follows
    Anscombe 1948, "The Transformation of Poisson, Binomial and Negative-Binomial Data"
    https://www.jstor.org/stable/2332343

    Subtility: Inverse uses -1/8, not -3/8
    Decent for high-count poisson denoising.

    :param x:
    :return:
    """
    return (x / 2) ** 2 - 1 / 8


def exact_unbiased_poisson_inverseVST(x):
    """
    Inverse follows
    M. Mäkitalo and A. Foi,
    "A closed-form approximation of the exact unbiased inverse of the Anscombe variance-stabilizing transformation",
    IEEE Trans. Image Process., vol. 20, no. 9, pp. 2697-2698, September 2011. doi:10.1109/TIP.2011.2121085

    Decent for low-count and high-count poisson denoising.

    Expression source: https://github.com/yinhaoz/denoising-fluorescence/blob/d83cad96b205793da0f62f6dc8094799d61929e6/denoising/matlab/_2_VST-BM3D_/code/Anscombe_inverse_closed_form.m#L20

    :param img:
    :return:
    """
    return (
        (x / 2) ** 2
        + 1 / 4 * sqrt(3 / 2) / x
        - 11 / 8 * torch.power(x, -2)
        + 5 / 8 * sqrt(3 / 2) * torch.power(x, -3)
        - 1 / 8
    )


def binomial_vst(x, n: int | Tensor, a: float = 0.77):
    """
    Variance stabilizing transform for binomial.yaml B(n,p)

    https://dornsife.usc.edu/sergey-lototsky/wp-content/uploads/sites/211/2023/06/VarianceStabilizationP-B-Nb.pdf

    :param x:
    :param n: n in B(n,p)
    :return:
    """
    if isinstance(n, int):
        n = torch.tensor(n)
    mul = torch.sqrt(n + 0.5)
    return mul * torch.arcsin((2 * x - n) / (n + a))


def binomial_inverseVST(x, n: int | Tensor, a: float = 0.77):
    """
    Inverse variance stabilizing transform for binomial.yaml B(n,p)

    Approximate variance is 1

    https://dornsife.usc.edu/assets/sites/1185/docs/others/VarianceStabilizationP-B-Nb.pdf

    :param x:
    :param n: n in B(n,p)
    :return:
    """
    if isinstance(n, int):
        n = torch.tensor(n)
    div = torch.sqrt(n + 0.5)
    out = torch.sin(x / div)
    out = (n + a) * out + n
    return out / 2


def floor_multiple_of(x: int, base: int = 5) -> int:
    return base * floor(x / base)


def create_1d_gaussian_kernels(
    kernel_size: int,
    sigmas: Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """
    Creates banks of 1D Gaussian and its second derivative kernels.
    """
    if not (kernel_size % 2 == 1):
        raise ValueError("Kernel size must be odd.")

    coords = torch.linspace(
        -kernel_size // 2, kernel_size // 2, kernel_size, device=device, dtype=dtype
    )
    coords = coords.view(1, kernel_size)
    sigmas_2 = (sigmas**2).view(sigmas.shape[0], 1)

    g_kernels = torch.exp(-(coords**2) / (2 * sigmas_2))
    g_kernels /= g_kernels.sum(dim=-1, keepdim=True)

    g_tt_kernels = (coords**2 - sigmas_2) / (sigmas_2**2) * g_kernels
    g_tt_kernels -= g_tt_kernels.mean(dim=-1, keepdim=True)

    return g_kernels, g_tt_kernels

