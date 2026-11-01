"""End-to-end smoke tests on the (downloaded) measuring_tape clip.

Fetch it with ``uv run python scripts/download_scenes.py measuring_tape``.
"""

import json
import time

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from probabilistic_events.data import REPO_ROOT, PhotonCube
from probabilistic_events.run import run

SCENE = REPO_ROOT / "data" / "scenes" / "measuring_tape.npy"
pytestmark = pytest.mark.skipif(not SCENE.exists(), reason=f"{SCENE} not downloaded")


def compose_config(name: str, overrides: list[str]):
    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base="1.3"):
        return compose(config_name=name, overrides=overrides)


def test_photon_cube_counts():
    cube = PhotonCube(SCENE, sum_frames=31, chunk_size=248, hot_pixel_mask=None)
    chunk = cube[0]
    assert chunk.shape == (256, 512, 8)
    assert 0 <= int(chunk.min()) and int(chunk.max()) <= 31
    packed = np.load(SCENE, mmap_mode="r")[:31]
    np.testing.assert_array_equal(chunk[..., 0].numpy(), np.unpackbits(packed, axis=-1).sum(0))


@pytest.mark.parametrize("method", ["default", "binomial"])
def test_run_cli(tmp_path, method):
    # 4 output frames x 8 exposures x 31 binary frames = 992 binary frames
    cfg = compose_config(
        "run", ["scene=measuring_tape", f"method={method}", "max_frames=4", "num_workers=0"]
    )
    stats = run(cfg, tmp_path)

    assert stats["frames"] == 4 and stats["binary_frames"] == 992
    for name in ("flux.mp4", "sample_weight.mp4", "events.mp4", "activity.png", "config.yaml"):
        assert (tmp_path / name).stat().st_size > 0, name
    assert json.loads((tmp_path / "stats.json").read_text())["frames"] == 4

    arrays = np.load(tmp_path / "arrays.npz")
    assert arrays["flux"].shape == (4, 256, 512)
    assert arrays["events"].dtype == np.int8 and set(np.unique(arrays["events"])) <= {-1, 0, 1}
    assert np.isfinite(arrays["flux"].astype(np.float32)).all()
    assert 0 < arrays["activity"].mean() < 1


def test_dashboard_serves_frames():
    from fastapi.testclient import TestClient

    from probabilistic_events.dashboard.server import create_app

    cfg = compose_config("dashboard", ["scene=measuring_tape", "method=binomial"])
    with TestClient(create_app(cfg)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/status").json()["scene"] == "measuring_tape"
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "seek", "index": 2})
            deadline = time.time() + 300
            while time.time() < deadline:
                msg = ws.receive_json()
                assert not msg.get("error"), msg
                if "frames" in msg and msg["index"] == 2:
                    break
            else:
                pytest.fail("no frame received")
        assert set(msg["frames"]) == {"flux", "sample_weight", "gradient", "events"}
        jpeg = client.get("/api/frame/flux.jpg")
        assert jpeg.status_code == 200 and jpeg.content[:2] == b"\xff\xd8"
