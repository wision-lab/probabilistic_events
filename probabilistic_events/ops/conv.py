from jaxtyping import Float
from torch import Tensor
from torch.nn import functional as F


def separable_conv(
    image: Float[Tensor, "n c h w"],
    kernel_y: Tensor,
    kernel_x: Tensor,
    dilation: int = 1,
) -> Tensor:
    """
    Performs a 2D convolution using two separable 1D kernels with dilation.
    """
    device, dtype = image.device, image.dtype
    num_channels = image.shape[1]

    ky = (
        kernel_y.to(device=device, dtype=dtype)
        .view(1, 1, -1, 1)
        .repeat(num_channels, 1, 1, 1)
    )
    kx = (
        kernel_x.to(device=device, dtype=dtype)
        .view(1, 1, 1, -1)
        .repeat(num_channels, 1, 1, 1)
    )

    # Calculate padding based on kernel size and dilation rate
    # Effective kernel size = (kernel_size - 1) * dilation + 1
    # Required padding = floor(effective_kernel_size / 2)
    pad_y = (ky.shape[2] - 1) // 2 * dilation
    pad_x = (kx.shape[3] - 1) // 2 * dilation

    response = F.conv2d(
        F.pad(image, (0, 0, pad_y, pad_y), mode="replicate"),
        ky,
        dilation=(dilation, 1),
        groups=num_channels,
    )
    response = F.conv2d(
        F.pad(response, (pad_x, pad_x, 0, 0), mode="replicate"),
        kx,
        dilation=(1, dilation),
        groups=num_channels,
    )
    return response
