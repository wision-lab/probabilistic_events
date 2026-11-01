import torch
from jaxtyping import Float
from torch import nn, Tensor


class GrayWorldWB(nn.Module):
    """
    Applies the "Gray World" white balancing algorithm.

    This algorithm assumes that the average color of the scene is gray.
    It computes the mean of each channel (R, G, B), calculates the
    mean of those means (the "gray" value), and then computes gains
    to scale each channel to match that gray value.
    """

    def __init__(self, gain_clip_min: float = 0.25, gain_clip_max: float = 4.0):
        super().__init__()
        self.gain_clip_min = gain_clip_min
        self.gain_clip_max = gain_clip_max
        self.epsilon = 1e-9

    @torch.compile()
    def forward(
        self, img: Float[Tensor, "batch channel height width"]
    ) -> Float[Tensor, "batch channel height width"]:
        """
        :param img: (B, C, H, W) input image tensor.
        :return: (B, C, H, W) white-balanced image tensor.
        """
        # Calculate the mean of each channel across batch, height, and width
        # Dims 0, 2, 3 -> (channel,)
        channel_means = torch.mean(img, dim=(0, 2, 3))

        # Calculate the overall "gray" mean
        gray_mean = torch.mean(channel_means)

        # Calculate gains, adding epsilon to prevent division by zero
        # (channel,)
        gains = gray_mean / (channel_means + self.epsilon)

        # Clip gains to a reasonable range to avoid extreme color casts
        gains = torch.clamp(gains, self.gain_clip_min, self.gain_clip_max)

        # Apply gains (B, C, H, W) * (1, C, 1, 1)
        return img * gains.view(1, -1, 1, 1)
