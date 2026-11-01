"""Loader helpers that don't need downloaded data."""

import numpy as np
import pytest
import torch

from probabilistic_events.data import bayer_nearest_neighbor_indices, transform_bayer_pattern


@pytest.mark.parametrize("pattern", ["rggb", "bggr", "grbg", "gbrg"])
@pytest.mark.parametrize("flip_ud", [False, True])
@pytest.mark.parametrize("k_rot", [0, 1, 2, 3])
def test_bayer_pattern_follows_frame_transform(pattern, flip_ud, k_rot):
    """The reported pattern must match a mosaic flipped/rotated like the frames."""
    tile = np.array(list(pattern)).reshape(2, 2)
    mosaic = torch.from_numpy(np.tile(np.vectorize(ord)(tile), (8, 8)))  # 16 x 16 CFA labels
    if flip_ud:
        mosaic = mosaic.flip(0)
    mosaic = torch.rot90(mosaic, k=k_rot, dims=(0, 1))
    top_left = "".join(chr(int(c)) for c in mosaic[:2, :2].flatten())
    assert transform_bayer_pattern(pattern, flip_ud, k_rot) == top_left


def test_bayer_hot_pixels_use_same_colour():
    rng = np.random.default_rng(0)
    mask = (rng.random((32, 40)) < 0.05).astype(np.uint8)
    near_i, near_j = bayer_nearest_neighbor_indices(mask)
    query_i, query_j = np.where(mask.astype(bool))
    assert (near_i % 2 == query_i % 2).all() and (near_j % 2 == query_j % 2).all()
    assert not mask[near_i, near_j].any()
