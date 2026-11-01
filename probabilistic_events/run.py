"""Run probabilistic events on a scene and write the outputs.

    uv run python -m probabilistic_events.run scene=newton_cradle
    uv run python -m probabilistic_events.run scene=tennis method=fast method.bocpd_gamma=1e-4

Writes to ``outputs/<scene>/<timestamp>/`` (see ``conf/run.yaml``):

    flux.mp4 / sample_weight.mp4 / events.mp4   videos of the three streams
    arrays.npz       flux, sample_weight (float16), events (int8) stacks, activity map
    activity.png     mean sample weight over the clip (colour-mapped)
    frames/<stream>/%06d.png                    if output.frames=true
    config.yaml, stats.json
"""

import json
import time
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from probabilistic_events.data import REPO_ROOT
from probabilistic_events.pipeline import (
    Frame,
    SAMPLE_WEIGHT_CMAP,
    build_dataset,
    build_model,
    select_device,
    step,
    to_rgb8,
)
from probabilistic_events.utils.plotting import apply_colormap_torch

STREAMS = ("flux", "sample_weight", "events")


def run(cfg: DictConfig | dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Run the method over ``cfg.scene`` and write outputs to ``out_dir``."""
    cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    method = OmegaConf.to_container(cfg.method, resolve=True)
    scene = OmegaConf.to_container(cfg.scene, resolve=True)
    out = cfg.output

    device = select_device(cfg.device)
    dataset = build_dataset(scene, method, cfg.exposures_per_frame)
    num_frames = len(dataset) if cfg.max_frames < 0 else min(cfg.max_frames, len(dataset))
    if num_frames == 0:
        raise ValueError(f"{dataset.file} is too short for one output frame")
    model = build_model(method, device, dataset)
    loader = DataLoader(
        torch.utils.data.Subset(dataset, range(num_frames)),
        batch_size=None,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    h, w = dataset.shape
    logger.info(
        f"{scene['name']}: {num_frames} frames of {dataset.chunk_size} binary frames "
        f"({dataset.t_out} x {dataset.sum_frames}), {h}x{w}, {device}, writing to {out_dir}"
    )
    OmegaConf.save(cfg, out_dir / "config.yaml")
    sink = logger.add(out_dir / "run.log", mode="w")

    writers = {}
    if out.video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writers = {s: cv2.VideoWriter(str(out_dir / f"{s}.mp4"), fourcc, out.fps, (w, h)) for s in STREAMS}
    if out.frames:
        for s in STREAMS:
            (out_dir / "frames" / s).mkdir(parents=True, exist_ok=True)
    arrays: dict[str, list[np.ndarray]] = {s: [] for s in STREAMS}
    activity = torch.zeros((h, w), device=device)
    event_counts = torch.zeros((2,), dtype=torch.long, device=device)

    start = time.perf_counter()
    with torch.no_grad():
        for index, cube in enumerate(tqdm(loader, total=num_frames, unit="frame")):
            frame: Frame = step(model, cube.to(device, non_blocking=True), index)
            activity += frame.sample_weight.float()
            event_counts += torch.stack([(frame.events > 0).sum(), (frame.events < 0).sum()])
            if out.arrays:
                arrays["flux"].append(frame.flux.half().cpu().numpy())
                arrays["sample_weight"].append(frame.sample_weight.half().cpu().numpy())
                arrays["events"].append(frame.events.to(torch.int8).cpu().numpy())
            if writers or out.frames:
                for name, image in to_rgb8(frame).items():
                    if name not in STREAMS:
                        continue
                    bgr = np.ascontiguousarray(image[..., ::-1])
                    if name in writers:
                        writers[name].write(bgr)
                    if out.frames:
                        cv2.imwrite(str(out_dir / "frames" / name / f"{index:06d}.png"), bgr)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    for writer in writers.values():
        writer.release()

    activity /= num_frames
    activity_rgb = apply_colormap_torch(activity.clamp(0, 1), SAMPLE_WEIGHT_CMAP).cpu().numpy()
    cv2.imwrite(str(out_dir / "activity.png"), activity_rgb[..., ::-1])
    if out.arrays:
        np.savez_compressed(
            out_dir / "arrays.npz",
            **{k: np.stack(v) for k, v in arrays.items()},
            activity=activity.cpu().numpy(),
        )

    stats = {
        "scene": scene["name"],
        "frames": num_frames,
        "binary_frames": num_frames * dataset.chunk_size,
        "seconds": round(elapsed, 3),
        "binary_fps": round(num_frames * dataset.chunk_size / elapsed, 1),
        "output_fps": round(num_frames / elapsed, 2),
        "positive_events": int(event_counts[0]),
        "negative_events": int(event_counts[1]),
        "mean_activity": float(activity.mean()),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(f"Done: {stats}")
    logger.remove(sink)
    return stats


@hydra.main(config_path=str(REPO_ROOT / "conf"), config_name="run", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run(cfg, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
