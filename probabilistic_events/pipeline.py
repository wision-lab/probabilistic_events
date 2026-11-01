"""Glue shared by the CLI (``run.py``) and the dashboard: model, data, colouring."""

from typing import Any, Mapping, NamedTuple

import numpy as np
import torch
from torch import Tensor

from probabilistic_events.data import PhotonCube
from probabilistic_events.integrator import ProbabilisticEvents
from probabilistic_events.utils.plotting import apply_colormap_torch, register_custom_cmaps

SAMPLE_WEIGHT_CMAP = "inferno_r"
EVENT_CMAP = "sparse_purple_green"
register_custom_cmaps()


class Frame(NamedTuple):
    index: int  # output frame (= chunk) index
    flux: Tensor  # (h, w) or (h, w, 3), [0, 1]
    sample_weight: Tensor  # (h, w), [0, 1]: omega_t, high = short exposure / activity
    events: Tensor  # (h, w) int: -1 resolution, 0 none, +1 surprise
    gradient: Tensor  # (h, w), [0, 1]: gradient magnitude of the flux (for display)


def select_device(device: str = "auto") -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def build_model(method: Mapping[str, Any], device: torch.device, dataset: PhotonCube | None = None) -> ProbabilisticEvents:
    """The model for ``method``; a Bayer ``dataset`` sets its CFA pattern (in output orientation)."""
    params = dict(method)
    params["tonemap"] = params.get("tonemap") or ""
    if dataset is not None and dataset.is_bayer:
        params["bayer_pattern"] = dataset.bayer_pattern
    return ProbabilisticEvents(**params).to(device)


def build_dataset(
    scene: Mapping[str, Any], method: Mapping[str, Any], exposures_per_frame: int, **overrides
) -> PhotonCube:
    """One dataset item = one output frame = ``exposures_per_frame`` virtual exposures."""
    sum_frames = int(method["sum_frames"])
    return PhotonCube.from_config(
        scene, sum_frames=sum_frames, chunk_size=exposures_per_frame * sum_frames, **overrides
    )


def step(model: ProbabilisticEvents, cube: Tensor, index: int) -> Frame:
    """Feed one chunk (h, w, t) of virtual exposures; return the state after it."""
    flux, gradient, sample_weight, events = model(cube)
    return Frame(index, flux, sample_weight, events, gradient)


def colormap_stops(name: str, n: int = 11) -> list[str]:
    """Hex colours sampled from a colormap at 0..1, for legends."""
    values = torch.linspace(0, 1, n).reshape(1, n)
    rgb = apply_colormap_torch(values, name).reshape(n, 3).tolist()
    return ["#%02x%02x%02x" % tuple(int(c) for c in px) for px in rgb]


def to_rgb8(frame: Frame) -> dict[str, np.ndarray]:
    """Colourise a frame's maps into (h, w, 3) uint8 images."""
    flux = frame.flux.float().clamp(0, 1)
    if flux.ndim == 2:
        flux = flux[..., None].expand(*flux.shape, 3)
    images = {
        "flux": (flux * 255).round().to(torch.uint8),
        "sample_weight": apply_colormap_torch(frame.sample_weight.float().clamp(0, 1), SAMPLE_WEIGHT_CMAP),
        "events": apply_colormap_torch((frame.events.float() + 1) / 2, EVENT_CMAP),
        "gradient": (frame.gradient.float().clamp(0, 1)[..., None].expand(-1, -1, 3) * 255)
        .round()
        .to(torch.uint8),
    }
    return {k: v.cpu().numpy() for k, v in images.items()}
