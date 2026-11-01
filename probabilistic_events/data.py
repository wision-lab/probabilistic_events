"""Loading SPAD photon cubes (``data/scenes/<scene>.npy`` or ``.h5``).

Two formats are supported, both written by ``scripts/download_scenes.py``:

- ``.npy``: a bit-packed binary photon cube of shape ``(t, h, w // 8)``, uint8
  (SwissSPAD2 and colorSPAD recordings).
- ``.h5``: a UbiCam recording (Ubicept ``.ubi.h5``), whose frames the camera has
  already summed over a fixed number of binary frames; they are summed further
  when ``sum_frames`` is a multiple of that number.

:class:`PhotonCube` slices a recording into chunks of multi-bit "virtual
exposures", corrects hot pixels (from the nearest pixel of the same colour on
Bayer sensors) and, for the colorSPAD prototype, crops the array and inpaints
its sparse colour-filtered pixels from their (unfiltered) neighbours.
"""

import json
from math import floor
from pathlib import Path
from typing import Any, Mapping

import cv2
import h5py
import numpy as np
import torch
from einops import rearrange, reduce
from jaxtyping import Int
from scipy import ndimage
from torch import Tensor
from torch.utils.data import Dataset

from probabilistic_events.ops.image import nearest_neighbor_indices

REPO_ROOT = Path(__file__).resolve().parents[1]

# colorSPAD readout is 256 x 512; the usable, rearranged area is 254 x 496.
COLOR_SPAD_SHAPE = (254, 496)


def resolve_path(path: str | Path | None) -> Path | None:
    """Resolve config paths relative to the repository root."""
    if path is None:
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def bayer_nearest_neighbor_indices(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`nearest_neighbor_indices`, but only from pixels of the same 2x2 CFA phase."""
    query = np.where(mask.astype(bool))
    near_i = np.empty(len(query[0]), dtype=np.int64)
    near_j = np.empty(len(query[0]), dtype=np.int64)
    for di in (0, 1):
        for dj in (0, 1):
            sub = mask[di::2, dj::2]
            if not sub.any():
                continue
            si, sj = nearest_neighbor_indices(sub)
            # Bad pixels of this phase, in the same order as np.where on the full mask
            sel = (query[0] % 2 == di) & (query[1] % 2 == dj)
            near_i[sel] = si * 2 + di
            near_j[sel] = sj * 2 + dj
    return near_i, near_j


def transform_bayer_pattern(pattern: str, flip_ud: bool, k_rot: int) -> str:
    """CFA pattern after the same flip and ``torch.rot90(k_rot)`` applied to the frames."""
    tile = np.array(list(pattern.lower())).reshape(2, 2)
    tile = np.tile(tile, (2, 2))  # a 4x4 tile keeps the top-left 2x2 well defined
    if flip_ud:
        tile = tile[::-1]
    tile = np.rot90(tile, k=k_rot, axes=(0, 1))
    return "".join(tile[:2, :2].ravel())


def read_ubi_metadata(file: Path) -> dict[str, Any]:
    """Frame group, frame keys, CFA pattern and hardware accumulation of a ``.ubi.h5`` file."""
    with h5py.File(file, "r") as f:
        found = []
        f.visititems(lambda name, obj: found.append(name) if not found and isinstance(obj, h5py.Dataset) and Path(name).name.isdigit() else None)
        if not found:
            raise ValueError(f"No frames found in {file}")
        group_path = str(Path(found[0]).parent)
        keys = sorted(f[group_path].keys(), key=int)
        meta = json.loads(f[found[0]].attrs.get("metadata", "{}"))
    poly = meta.get("polycam_metadata", {})
    return {
        "group": group_path,
        # The first frame of a capture is a partial integration; skip it
        "keys": keys[1:],
        "bayer_pattern": str(meta.get("cfa_pattern", "gray")).lower(),
        "frames_per_count": int(poly.get("num_integrated_binary_frames", meta.get("num_integrated_binary_frames", 1))),
    }


def interpolate_filtered_pixels(
    photon_cube: np.ndarray,
    sum_frames: int,
    cfa_mask: np.ndarray,
    hot_pixel_mask: np.ndarray | None,
) -> np.ndarray:
    """Replace colour-filtered pixels with binomial draws at their neighbours' rate."""
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    valid = np.ones(photon_cube.shape[:2], dtype=np.float32)
    if hot_pixel_mask is not None:
        valid = (~hot_pixel_mask.astype(bool)).astype(np.float32)
    neighbor_counts = ndimage.convolve(valid, kernel, mode="constant", cval=0)
    neighbor_sums = ndimage.convolve(
        photon_cube * valid[..., None], kernel[..., None], mode="constant", cval=0
    )
    neighbor_avg = np.divide(
        neighbor_sums,
        neighbor_counts[..., None],
        out=np.zeros_like(neighbor_sums, dtype=np.float32),
        where=neighbor_counts[..., None] != 0,
    )
    new_values = np.random.binomial(sum_frames, np.clip(neighbor_avg / sum_frames, 0, 1))
    return np.where(cfa_mask[..., None], new_values, photon_cube)


def color_spad_correct(
    photon_cube: np.ndarray,
    sum_frames: int,
    cfa_mask: np.ndarray,
    hot_pixel_mask: np.ndarray | None,
) -> np.ndarray:
    """Crop/rearrange the colorSPAD readout and inpaint its colour-filtered pixels."""
    cropped = np.zeros((*COLOR_SPAD_SHAPE, photon_cube.shape[-1]), dtype=photon_cube.dtype)
    cropped[:, :496] = photon_cube[2:, :496]
    cropped[:, 252:256] = photon_cube[2:, 260:264]
    cropped[:, 260:264] = photon_cube[2:, 252:256]
    return interpolate_filtered_pixels(
        cropped.astype(np.int64), sum_frames, cfa_mask, hot_pixel_mask
    )


class PhotonCube(Dataset):
    """Chunks of summed binary frames from a ``.npy`` photon cube or ``.h5`` UbiCam recording.

    Each item is an integer tensor of shape ``(h, w, chunk_size // sum_frames)``
    holding photon counts in ``[0, sum_frames]``. Time arguments are in binary
    frames for both formats.

    Args:
        file: bit-packed photon cube, ``(t, h, w // 8)`` uint8, or ``.ubi.h5``.
        sum_frames: binary frames summed into one virtual exposure.
        chunk_size: binary frames per item (a multiple of ``sum_frames``).
        initial_time_step: first binary frame to use.
        num_time_step: number of binary frames to use (-1: all).
        hot_pixel_mask: optional ``.npy`` mask of bad pixels (1 = bad).
        color_spad_cfa: RGBW CFA ``.tif`` of the colorSPAD; enables its correction.
        rotation: counter-clockwise rotation in degrees (0, 90, 180, 270).
        flip_ud: flip vertically (applied before the rotation).
        bayer_pattern: CFA of the raw frames (``gray``, ``rggb``, ...); ``.h5``
            files default to the pattern in their metadata. :attr:`bayer_pattern`
            gives the pattern after flip and rotation, for the model.
    """

    def __init__(
        self,
        file: str | Path,
        sum_frames: int,
        chunk_size: int,
        initial_time_step: int = 0,
        num_time_step: int = -1,
        hot_pixel_mask: str | Path | None = None,
        color_spad_cfa: str | Path | None = None,
        rotation: int = 0,
        flip_ud: bool = False,
        bayer_pattern: str | None = None,
    ):
        if chunk_size % sum_frames:
            raise ValueError(f"chunk_size ({chunk_size}) must be a multiple of sum_frames ({sum_frames})")
        if rotation not in (0, 90, 180, 270):
            raise ValueError(f"rotation must be 0, 90, 180 or 270, got {rotation}")

        self.file = resolve_path(file)
        if not self.file.exists():
            raise FileNotFoundError(
                f"{self.file} not found; fetch it with `uv run python scripts/download_scenes.py`"
            )
        self.is_h5 = self.file.suffix == ".h5"
        if self.is_h5:
            ubi = read_ubi_metadata(self.file)
            self.frames_per_count = ubi["frames_per_count"]
            if sum_frames % self.frames_per_count:
                raise ValueError(
                    f"{self.file.name} stores counts of {self.frames_per_count} binary frames; "
                    f"sum_frames ({sum_frames}) must be a multiple of it"
                )
            self._h5_group, self._h5_keys = ubi["group"], ubi["keys"]
            self._h5 = None  # opened lazily, once per process (DataLoader workers)
            bayer_pattern = bayer_pattern or ubi["bayer_pattern"]
            with h5py.File(self.file, "r") as f:
                h, w = f[self._h5_group][self._h5_keys[0]].shape[:2]
            total = len(self._h5_keys) * self.frames_per_count
        else:
            self.cube = np.load(self.file, mmap_mode="r")
            total, h, w8 = self.cube.shape
            w = w8 * 8

        self.sum_frames = sum_frames
        self.chunk_size = chunk_size
        self.t_out = chunk_size // sum_frames
        self.k_rot = rotation // 90
        self.flip_ud = flip_ud
        self.initial_time_step = initial_time_step
        available = max(0, total - initial_time_step)
        self.num_time_step = available if num_time_step < 0 else min(num_time_step, available)
        self._len = floor(self.num_time_step / chunk_size)

        self.cfa_mask = None
        if color_spad_cfa is not None:
            cfa = cv2.imread(str(resolve_path(color_spad_cfa)))
            if cfa is None:
                raise FileNotFoundError(f"Could not read CFA {color_spad_cfa}")
            self.cfa_mask = cfa[2:, :496, ::-1].mean(axis=-1) < 255
            h, w = COLOR_SPAD_SHAPE
        self.raw_shape = (h, w)

        raw_pattern = (bayer_pattern or "gray").lower()
        self.is_bayer = raw_pattern not in ("gray", "grey")
        self.bayer_pattern = (
            transform_bayer_pattern(raw_pattern, flip_ud, self.k_rot) if self.is_bayer else "gray"
        )

        self.hot_pixel_mask = None
        if hot_pixel_mask is not None:
            mask = np.load(resolve_path(hot_pixel_mask)).astype(np.uint8)
            if mask.shape != self.raw_shape:
                raise ValueError(f"Hot pixel mask shape {mask.shape} != frame shape {self.raw_shape}")
            self.hot_pixel_mask = mask
            self.hp_query = np.where(mask.astype(bool))
            nearest = bayer_nearest_neighbor_indices if self.is_bayer else nearest_neighbor_indices
            self.hp_nearest = nearest(mask)

        self.dtype = torch.uint8 if sum_frames <= 255 else torch.int16

    @classmethod
    def from_config(cls, scene: Mapping[str, Any], sum_frames: int, chunk_size: int, **overrides):
        """Build from a ``conf/scene/*.yaml`` mapping."""
        kwargs = dict(
            file=scene["file"],
            initial_time_step=scene.get("initial_time_step", 0),
            num_time_step=scene.get("num_time_step", -1),
            hot_pixel_mask=scene.get("hot_pixel_mask"),
            color_spad_cfa=scene.get("color_spad_cfa"),
            rotation=scene.get("rotation", 0),
            flip_ud=scene.get("flip_ud", False),
            bayer_pattern=scene.get("bayer_pattern"),
        )
        kwargs.update(overrides)
        return cls(sum_frames=sum_frames, chunk_size=chunk_size, **kwargs)

    @property
    def shape(self) -> tuple[int, int]:
        """Output (h, w) after rotation."""
        h, w = self.raw_shape
        return (w, h) if self.k_rot % 2 else (h, w)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Int[Tensor, "h w t"]:
        if not 0 <= index < self._len:
            raise IndexError(index)
        start = self.initial_time_step + index * self.chunk_size
        if self.is_h5:
            counts = self._read_h5(start)
        else:
            packed = self.cube[start : start + self.chunk_size]
            binary = rearrange(np.unpackbits(packed, axis=-1), "t h w -> h w t")
            counts = reduce(binary, "h w (t s) -> h w t", "sum", s=self.sum_frames)

        if self.cfa_mask is not None:
            counts = color_spad_correct(counts, self.sum_frames, self.cfa_mask, self.hot_pixel_mask)
        if self.hot_pixel_mask is not None:
            counts = counts.copy()
            counts[self.hp_query] = counts[self.hp_nearest]

        out = torch.from_numpy(np.ascontiguousarray(counts)).to(self.dtype)
        if self.flip_ud:
            out = out.flip(0)
        if self.k_rot:
            out = torch.rot90(out, k=self.k_rot, dims=(0, 1)).contiguous()
        return out

    def _read_h5(self, start: int) -> np.ndarray:
        """Counts for ``chunk_size`` binary frames from ``start``, as (h, w, t_out)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.file, "r")
        group = self._h5[self._h5_group]
        first = start // self.frames_per_count
        num = self.chunk_size // self.frames_per_count
        frames = np.stack(
            [group[key][...].reshape(self.raw_shape) for key in self._h5_keys[first : first + num]], axis=-1
        )
        per_exposure = self.sum_frames // self.frames_per_count
        return reduce(frames.astype(np.int16), "h w (t s) -> h w t", "sum", s=per_exposure)

    def __getstate__(self):
        # h5py handles can't be pickled (DataLoader workers); reopen on first read
        state = self.__dict__.copy()
        state["_h5"] = None
        return state
