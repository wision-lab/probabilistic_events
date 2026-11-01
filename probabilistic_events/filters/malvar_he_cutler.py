from typing import Literal

import torch
from jaxtyping import Float
from torch import nn, Tensor, autocast
from torch.nn import functional as F

BayerPatternLiteral = Literal["gray", "grey", "rggb", "bggr", "grbg", "gbrg"]


class MalvarHeCutlerDemosaic(nn.Module):
    def __init__(
        self,
        bayer_pattern: BayerPatternLiteral = "rggb",
        wr: float = 0.2126,
        wg: float = 0.7152,
        wb: float = 0.0722,
    ):
        super().__init__()
        self.bayer_pattern = bayer_pattern.lower()
        self.epsilon = 1e-8
        self.padding = 2

        self.is_bayer = self.bayer_pattern not in ["gray", "grey"]

        if self.is_bayer:
            self.wr, self.wg, self.wb = wr, wg, wb

            k_g_at_rb = torch.tensor(
                [
                    [0, 0, -2, 0, 0],
                    [0, 0, 4, 0, 0],
                    [-2, 4, 8, 4, -2],
                    [0, 0, 4, 0, 0],
                    [0, 0, -2, 0, 0],
                ],
                dtype=torch.float,
            )
            k_rb_at_gr = torch.tensor(
                [
                    [0, 0, 1, 0, 0],
                    [0, -2, 0, -2, 0],
                    [-2, 8, 10, 8, -2],
                    [0, -2, 0, -2, 0],
                    [0, 0, 1, 0, 0],
                ],
                dtype=torch.float,
            )
            k_rb_at_gb = k_rb_at_gr.T
            k_rb_at_rb = torch.tensor(
                [
                    [0, 0, -3, 0, 0],
                    [0, 4, 0, 4, 0],
                    [-3, 0, 12, 0, -3],
                    [0, 4, 0, 4, 0],
                    [0, 0, -3, 0, 0],
                ],
                dtype=torch.float,
            )
            k_delta = torch.zeros((5, 5), dtype=torch.float)
            k_delta[2, 2] = 16.0

            kernels_rgb = (
                torch.stack([k_g_at_rb, k_rb_at_gr, k_rb_at_gb, k_rb_at_rb]) / 16.0
            )
            self.register_buffer("kernels_rgb", kernels_rgb.view(4, 1, 5, 5))

            k_y_r = self.wr * k_delta + self.wg * k_g_at_rb + self.wb * k_rb_at_rb
            k_y_gr = self.wr * k_rb_at_gr + self.wg * k_delta + self.wb * k_rb_at_gb
            k_y_gb = self.wr * k_rb_at_gb + self.wg * k_delta + self.wb * k_rb_at_gr
            k_y_b = self.wr * k_rb_at_rb + self.wg * k_g_at_rb + self.wb * k_delta

            kernels_y = torch.stack([k_y_r, k_y_gr, k_y_gb, k_y_b]) / 16.0
            self.register_buffer("kernels_y", kernels_y.view(4, 1, 5, 5))

            rggb_rgb_index = torch.tensor(
                [
                    [[4, 1], [2, 3]],
                    [[0, 4], [4, 0]],
                    [[3, 2], [1, 4]],
                ],
                dtype=torch.long,
            ).view(1, 3, 2, 2)

            rggb_y_index = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long).view(
                1, 1, 2, 2
            )

            norm_y_r = torch.sqrt(torch.sum(k_y_r**2))
            norm_y_gr = torch.sqrt(torch.sum(k_y_gr**2))
            norm_y_gb = torch.sqrt(torch.sum(k_y_gb**2))
            norm_y_b = torch.sqrt(torch.sum(k_y_b**2))

            rggb_norm_map = (
                torch.tensor(
                    [[[norm_y_r, norm_y_gr], [norm_y_gb, norm_y_b]]],
                    dtype=torch.float,
                )
                / 16.0
            ).view(1, 1, 2, 2)

            if self.bayer_pattern == "rggb":
                self.register_buffer("index_rgb", rggb_rgb_index)
                self.register_buffer("index_y", rggb_y_index)
                self.register_buffer("norm_y_map", rggb_norm_map)
            elif self.bayer_pattern == "bggr":
                self.register_buffer(
                    "index_rgb", torch.roll(rggb_rgb_index, (1, 1), (-1, -2))
                )
                self.register_buffer(
                    "index_y", torch.roll(rggb_y_index, (1, 1), (-1, -2))
                )
                self.register_buffer(
                    "norm_y_map", torch.roll(rggb_norm_map, (1, 1), (-1, -2))
                )
            elif self.bayer_pattern == "grbg":
                self.register_buffer("index_rgb", torch.roll(rggb_rgb_index, 1, -1))
                self.register_buffer("index_y", torch.roll(rggb_y_index, 1, -1))
                self.register_buffer("norm_y_map", torch.roll(rggb_norm_map, 1, -1))
            elif self.bayer_pattern == "gbrg":
                self.register_buffer("index_rgb", torch.roll(rggb_rgb_index, 1, -2))
                self.register_buffer("index_y", torch.roll(rggb_y_index, 1, -2))
                self.register_buffer("norm_y_map", torch.roll(rggb_norm_map, 1, -2))

        self.register_buffer("cached_index_rgb", None, persistent=False)
        self.register_buffer("cached_index_y", None, persistent=False)
        self.register_buffer("cached_norm_map", None, persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.cached_index_rgb = None
        self.cached_index_y = None
        self.cached_norm_map = None
        return self

    def _get_tiled_map(
        self,
        ref_tensor: Tensor,
        map_buffer: Tensor,
        cache_name: str,
    ) -> Tensor:
        b, _, h, w = ref_tensor.shape
        cached_map = getattr(self, cache_name)

        if (
            cached_map is not None
            and cached_map.shape == (b, map_buffer.shape[1], h, w)
            and cached_map.device == ref_tensor.device
        ):
            return cached_map

        h_rep = (h + 1) // 2
        w_rep = (w + 1) // 2

        tiled_map = map_buffer.repeat(1, 1, h_rep, w_rep)
        tiled_map = tiled_map[..., :h, :w]
        tiled_map = tiled_map.expand(b, -1, -1, -1).to(device=ref_tensor.device)

        setattr(self, cache_name, tiled_map)
        return tiled_map

    @torch.compile()
    @autocast(device_type="cuda", enabled=False)
    def demosaic_linear(
        self, img: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, "batch 3 height width"]:
        if not self.is_bayer:
            return img.repeat(1, 3, 1, 1)

        img_f32 = img.float()
        xpad = F.pad(img_f32, (self.padding,) * 4, mode="replicate")
        planes = F.conv2d(xpad, self.kernels_rgb, padding=0)
        planes = torch.cat([planes, img_f32], dim=1)

        index_map = self._get_tiled_map(img_f32, self.index_rgb, "cached_index_rgb")
        rgb = torch.gather(planes, 1, index_map)

        return rgb.to(img.dtype)

    @torch.compile()
    @autocast(device_type="cuda", enabled=False)
    def forward(
        self, img: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, "batch 1 height width"]:
        if not self.is_bayer:
            return img

        img_f32 = img.float()
        xpad = F.pad(img_f32, (self.padding,) * 4, mode="replicate")
        planes_y = F.conv2d(xpad, self.kernels_y, padding=0)

        index_map = self._get_tiled_map(img_f32, self.index_y, "cached_index_y")
        y_unnorm = torch.gather(planes_y, 1, index_map)

        norm_map = self._get_tiled_map(img_f32, self.norm_y_map, "cached_norm_map")
        return (y_unnorm / (norm_map + self.epsilon)).to(img.dtype)
