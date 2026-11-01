"""Live dashboard: play a scene through ProbabilisticEvents in the browser.

    uv run python -m probabilistic_events.dashboard                  # casino_roulette
    uv run python -m probabilistic_events.dashboard scene=tennis port=8080

One shared :class:`Player` owns the model on the GPU and a worker thread that
advances it; every connected browser sees the same stream. Clients talk to it
over ``/ws`` (JSON commands in, JSON status + base64 JPEG frames out). The
latest frame is also served at ``/api/frame/<stream>.jpg`` for scripting.
"""

import asyncio
import base64
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import hydra
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from probabilistic_events.data import REPO_ROOT, PhotonCube
from probabilistic_events.pipeline import (
    EVENT_CMAP,
    SAMPLE_WEIGHT_CMAP,
    build_dataset,
    build_model,
    colormap_stops,
    select_device,
    step,
    to_rgb8,
)

CONF_DIR = REPO_ROOT / "conf"
WEB_DIR = Path(__file__).resolve().parent / "web"
STREAMS = ("flux", "sample_weight", "gradient", "events")
def _optional_int(value: Any) -> int | None:
    """Integer, with 0 / empty meaning None (e.g. no PCA projection)."""
    return None if value in (None, "", 0, "0") else int(value)


# Method parameters editable from the browser, with their types. "features" is
# a pseudo-parameter that sets the (mutually exclusive) model_* feature flags.
TUNABLE = {
    "bocpd_gamma": float,
    "memory_size": int,
    "sum_frames": int,
    "model_binomial": bool,
    "use_bayesian_predictive": bool,
    "features": str,
    "num_scales": int,
    "num_orientations": int,
    "pca_components": _optional_int,
    "max_filter_size": int,
    "denoise": bool,
    "denoise_window_size": int,
    "denoise_noise_gain": float,
    "linearize_response": bool,
    "flux_est_quantile": float,
    "white_balance": bool,
    "tonemap": str,
    "entropy_event_threshold": float,
    "backend": str,
}
BACKENDS = ("auto", "torch", "triton")
FEATURES = {
    "none": {},
    "gradient": {"model_gradient": True},
    "log_gabor": {"model_log_gabor": True},
    "difference_of_gaussians": {"model_difference_of_gaussians": True},
}
TONEMAPS = ("", "srgb", "ashikhmin", "reinhard_2002_global")


def features_of(method: dict[str, Any]) -> str:
    return next((k for k, flags in FEATURES.items() if flags and all(method.get(f) for f in flags)), "none")


def method_params(method: dict[str, Any]) -> dict[str, Any]:
    """The browser's view of a method config."""
    return {k: features_of(method) if k == "features" else method.get(k) for k in TUNABLE}
LEGENDS = {"sample_weight": colormap_stops(SAMPLE_WEIGHT_CMAP), "events": colormap_stops(EVENT_CMAP)}


def list_variants() -> list[str]:
    return sorted(p.stem for p in (CONF_DIR / "method").glob("*.yaml"))


def load_variant(name: str) -> dict[str, Any]:
    """conf/method/<name>.yaml on top of default.yaml (their only `defaults` entry)."""
    if name not in list_variants():
        raise ValueError(f"Unknown method {name!r}")
    cfg = OmegaConf.load(CONF_DIR / "method" / "default.yaml")
    overlay = OmegaConf.load(CONF_DIR / "method" / f"{name}.yaml")
    overlay.pop("defaults", None)
    return OmegaConf.to_container(OmegaConf.merge(cfg, overlay), resolve=True)


def list_scenes() -> list[str]:
    return sorted(p.stem for p in (CONF_DIR / "scene").glob("*.yaml"))


def load_scene(name: str) -> dict[str, Any]:
    if name not in list_scenes():
        raise ValueError(f"Unknown scene {name!r}")
    return OmegaConf.to_container(OmegaConf.load(CONF_DIR / "scene" / f"{name}.yaml"), resolve=True)


class Player:
    """Owns the dataset/model and a worker thread that produces frames."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = select_device(cfg.device)
        self.method: dict[str, Any] = OmegaConf.to_container(cfg.method, resolve=True)
        self.variant = self._match_variant(self.method)
        self.lock = threading.Condition()
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.prefetch: dict[int, Future] = {}
        self.dataset: PhotonCube | None = None
        self.model = None
        self.scene = ""
        self.playing = False
        self.position = 0  # index of the next frame to compute
        self.seek_to: int | None = None
        self.rebuild = False
        self.error = ""
        self.busy = ""  # what the worker is doing, if it may take a while
        self.compute_ms = 0.0
        self.frame: dict[str, Any] | None = None  # latest {index, jpegs}
        self.revision = 0
        self.load(cfg.scene.name)
        threading.Thread(target=self._worker, daemon=True, name="player").start()

    # -- commands (any thread) ----------------------------------------------
    def load(self, scene: str) -> None:
        dataset = build_dataset(load_scene(scene), self.method, self.cfg.exposures_per_frame)
        with self.lock:
            self.dataset, self.scene, self.prefetch = dataset, scene, {}
            # A new scene may have a different CFA (and size), so rebuild the model
            self.seek_to, self.error, self.rebuild = 0, "", self.model is not None
            self._changed()

    def set_params(self, params: dict[str, Any], variant: str | None = None) -> None:
        new = load_variant(variant) if variant else dict(self.method)
        for key, value in params.items():
            if key not in TUNABLE:
                raise ValueError(f"{key} is not tunable")
            value = TUNABLE[key](value)
            if key == "features":
                if value not in FEATURES:
                    raise ValueError(f"features must be one of {list(FEATURES)}")
                for flags in FEATURES.values():
                    for flag in flags:
                        new[flag] = False
                new.update(FEATURES[value])
            else:
                new[key] = value
        problems = [
            new["backend"] not in BACKENDS and "backend",
            new["tonemap"] not in TONEMAPS and "tonemap",
            not 0 < new["bocpd_gamma"] < 1 and "bocpd_gamma (0 < γ < 1)",
            new["sum_frames"] < 1 and "sum_frames",
            new["memory_size"] < 1 and "memory_size",
            min(new["num_scales"], new["num_orientations"], new["max_filter_size"]) < 1 and "filter sizes",
            new["denoise_window_size"] < 1 and "denoise_window_size",
            not 0 < new["flux_est_quantile"] <= 1 and "flux_est_quantile",
        ]
        problems = [p for p in problems if p]
        if problems:
            raise ValueError(f"Invalid {', '.join(problems)}")
        if not (new["model_binomial"] or features_of(new) != "none"):
            raise ValueError("Enable the binomial likelihood or a spatial feature model")
        with self.lock:
            index = self.frame["index"] if self.frame else 0
            old_chunk = self.dataset.chunk_size
            if new["sum_frames"] != self.method["sum_frames"]:
                self.dataset = build_dataset(load_scene(self.scene), new, self.cfg.exposures_per_frame)
                self.prefetch = {}
                index = index * old_chunk // self.dataset.chunk_size
            self.method = new
            self.variant = self._match_variant(new)
            self.rebuild, self.seek_to = True, min(index, len(self.dataset) - 1)
            self._changed()

    @staticmethod
    def _match_variant(method: dict[str, Any]) -> str:
        """The preset in conf/method/ these settings equal, else "custom"."""
        return next((v for v in list_variants() if load_variant(v) == method), "custom")

    def play(self, playing: bool) -> None:
        with self.lock:
            self.playing = playing
            self._changed()

    def seek(self, index: int) -> None:
        with self.lock:
            self.seek_to = max(0, min(int(index), len(self.dataset) - 1))
            self._changed()

    def status(self) -> dict[str, Any]:
        h, w = self.dataset.shape
        return {
            "type": "status",
            "scene": self.scene,
            "scenes": list_scenes(),
            "playing": self.playing,
            "index": self.frame["index"] if self.frame else None,
            "num_frames": len(self.dataset),
            "binary_frames_per_frame": self.dataset.chunk_size,
            "shape": [h, w],
            "params": method_params(self.method),
            "variant": self.variant,
            "variants": list_variants(),
            "legends": LEGENDS,
            "exposures_per_frame": self.cfg.exposures_per_frame,
            "compute_ms": round(self.compute_ms, 1),
            "error": self.error,
            "busy": self.busy,
            "revision": self.revision,
        }

    def wait(self, revision: int, timeout: float) -> int:
        with self.lock:
            self.lock.wait_for(lambda: self.revision != revision, timeout)
            return self.revision

    def _set_busy(self, message: str) -> None:
        with self.lock:
            self.busy = message
            self._changed()

    def _changed(self) -> None:
        self.revision += 1
        self.lock.notify_all()

    # -- worker thread ------------------------------------------------------
    def _chunk(self, dataset: PhotonCube, index: int) -> torch.Tensor:
        for i in range(index, min(index + 4, len(dataset))):
            if i not in self.prefetch:
                self.prefetch[i] = self.pool.submit(dataset.__getitem__, i)
        cube = self.prefetch.pop(index).result()
        for stale in [i for i in self.prefetch if i < index]:
            del self.prefetch[stale]
        return cube

    def _compute(self, dataset: PhotonCube, index: int):
        return step(self.model, self._chunk(dataset, index).to(self.device), index)

    @torch.no_grad()
    def _worker(self) -> None:
        min_period = 1.0 / self.cfg.max_fps
        while True:
            with self.lock:
                self.lock.wait_for(lambda: self.playing or self.seek_to is not None or self.rebuild)
                dataset, seek_to, rebuild = self.dataset, self.seek_to, self.rebuild
                self.seek_to, self.rebuild = None, False
            start = time.perf_counter()
            try:
                if rebuild or self.model is None:
                    self.model = build_model(self.method, self.device, dataset)
                    self._set_busy("Compiling the model (torch.compile; can take ~30 s)")
                elif seek_to is not None:
                    self._set_busy(f"Seeking (replaying {self.cfg.warm_up_frames} warm-up frames)")
                if seek_to is not None:
                    # The method is recursive: replay a short warm-up before the target.
                    self.model.reset()
                    for i in range(max(0, seek_to - self.cfg.warm_up_frames), seek_to):
                        self._compute(dataset, i)
                    self.position = seek_to
                elif self.position >= len(dataset):
                    self.model.reset()
                    self.position = 0
                frame = self._compute(dataset, self.position)
                rgb = to_rgb8(frame)
                params = [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.jpeg_quality)]
                jpegs = {k: cv2.imencode(".jpg", v[..., ::-1], params)[1].tobytes() for k, v in rgb.items()}
                with self.lock:
                    if dataset is self.dataset:
                        self.frame = {"index": frame.index, "jpegs": jpegs}
                        self.position = frame.index + 1
                        self.compute_ms = (time.perf_counter() - start) * 1e3
                        self.busy = self.error = ""
                        self._changed()
            except Exception as exc:  # surface errors in the UI rather than killing the thread
                logger.exception(exc)
                with self.lock:
                    self.playing, self.busy = False, ""
                    self.error = f"{type(exc).__name__}: {exc}"
                    self._changed()
            time.sleep(max(0.0, min_period - (time.perf_counter() - start)))


def create_app(cfg: DictConfig) -> FastAPI:
    app = FastAPI(title="Probabilistic events dashboard")
    player = Player(cfg)
    app.state.player = player
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.middleware("http")
    async def no_cache(request, call_next):
        # The UI is tiny; always revalidate so edits show up without a hard refresh
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return player.status()

    @app.get("/api/frame/{stream}.jpg")
    def frame(stream: str) -> Response:
        if stream not in STREAMS:
            raise HTTPException(404, f"stream must be one of {STREAMS}")
        if player.frame is None:
            raise HTTPException(503, "no frame computed yet")
        return Response(player.frame["jpegs"][stream], media_type="image/jpeg")

    async def handle(msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "load":
            await asyncio.to_thread(player.load, msg["scene"])
        elif kind == "params":
            await asyncio.to_thread(player.set_params, msg.get("params", {}), msg.get("variant"))
        elif kind == "play":
            player.play(bool(msg.get("playing", True)))
        elif kind == "seek":
            player.seek(msg["index"])
        else:
            raise ValueError(f"Unknown message type {kind!r}")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()

        async def receive() -> None:
            while True:
                msg = await websocket.receive_json()
                try:
                    await handle(msg)
                except Exception as exc:
                    await websocket.send_json({"type": "error", "error": f"{type(exc).__name__}: {exc}"})

        async def send() -> None:
            revision, sent = -1, None
            while True:
                revision = await asyncio.to_thread(player.wait, revision, 1.0)
                status = player.status()
                frame = player.frame
                if frame is not None and frame is not sent:
                    status["frames"] = {k: base64.b64encode(v).decode() for k, v in frame["jpegs"].items()}
                    sent = frame
                await websocket.send_json(status)

        tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        except WebSocketDisconnect:
            pass
        finally:
            for task in tasks:
                task.cancel()

    return app


@hydra.main(config_path=str(CONF_DIR), config_name="dashboard", version_base="1.3")
def main(cfg: DictConfig) -> None:
    app = create_app(cfg)
    logger.info(f"Dashboard on http://{cfg.host}:{cfg.port}")
    uvicorn.run(app, host=cfg.host, port=cfg.port, access_log=False)


if __name__ == "__main__":
    main()
