# Probabilistic Events

Code for **Quanta Perception as Probabilistic Events**
([arXiv:2608.27584](https://arxiv.org/abs/2608.27584)).

Probabilistic events turn a stream of single-photon (quanta) frames into a
recursive belief over the time since each pixel's intensity last changed. From
that run-length posterior we read out three low-latency signals: a
motion-adaptive flux estimate, an activity (sample-weight) map, and
entropy-based uncertainty events.

The code is released under the UW academic, non-commercial software license;
see [LICENSE](LICENSE).

## Installation

Install [uv](https://docs.astral.sh/uv/), then from this folder:

```bash
uv sync
```

This creates `.venv` with Python >=3.10 and PyTorch 2.6 (CUDA 12.4 wheels on
Linux/Windows; PyPI wheels on macOS). Prefix commands with `uv run`. Add
`--group dev` for the tests.

## Data

Download the example scenes (about 33 GB in total) and sensor masks:

```bash
uv run python scripts/download_scenes.py          # all scenes
uv run python scripts/download_scenes.py --list   # see what is available
```

Each config in `conf/scene/` names its source recording in the
[WISION Lab public datasets](https://web.s3.wisc.edu/public-datasets/) and the
recording inside it. SPAD recordings (2-3 GB each) are fetched on their own via
HTTP range requests into `data/scenes/<scene>.npy`. The two UbiCam night scenes
come from one 3.7 GB archive, which is streamed once and unpacked into
`data/scenes/<scene>.h5` (11 GB for both). Hot-pixel masks and the
colour-filter layout go under `data/`. Scenes start at the frame the paper uses
and run to the end of the recording; set `scene.num_time_step` to shorten them.

| Scene | Content |
|---|---|
| `casino_roulette` | Spinning roulette wheel (dashboard default) |
| `newton_cradle` | Newton's cradle (Fig. 3B) |
| `prize_wheel` | Rapidly spinning prize wheel (Fig. 3D) |
| `measuring_tape` | Snapping measuring tape |
| `capitol` | Hand-held capitol model |
| `traffic_night` | Night-time city traffic |
| `tennis` | Indoor tennis |
| `runner_low_light` | Person running at about 0.05 lux |
| `night_drive` | Night-time driving, 1 MPixel UbiCam in colour (Bayer) |
| `night_east_pause` | Night-time street toward the Wisconsin State Capitol, 1 MPixel UbiCam in colour |

The colorSPAD scenes (`capitol`, `casino_roulette`, `traffic_night`, `tennis`)
come out in greyscale: their sparse colour-filtered pixels are inpainted from
their neighbours. The UbiCam scenes are full Bayer mosaics and are demosaiced
into colour; they are dark, so add `method.tonemap=srgb`.

## Running

Process a scene and write videos and arrays to `outputs/<scene>/<timestamp>/`:

```bash
uv run python -m probabilistic_events.run scene=newton_cradle
```

Configs use [Hydra](https://hydra.cc/), so any value can be overridden on the
command line, e.g. `method=binomial`, `method.bocpd_gamma=1e-4` or
`max_frames=50`. Method variants live in `conf/method/`:

- `default`: binomial likelihood with Log-Gabor features (the robust variant)
- `fast`: binomial likelihood with image gradients
- `binomial`: binomial likelihood only (fastest)

All three run on the fused Triton kernel on CUDA.

Outputs are `flux.mp4`, `sample_weight.mp4`, `events.mp4`, `activity.png`,
`arrays.npz` and `stats.json`.

## Dashboard

```bash
uv run python -m probabilistic_events.dashboard   # then open http://127.0.0.1:8000
```

Choose a scene and a method (robust, fast or binomial), then play, pause and
seek through the reconstruction, sample weight, gradient magnitude and entropy
events. The sidebar exposes the method's parameters while it runs: γ, the
memory size, N_sum, the likelihood terms, the spatial features (type, scales,
orientations, PCA components), the denoiser (with an on/off toggle), the ISP
(linearisation, white balance, tonemap), the event threshold and the backend. Pass `scene=tennis` or `port=8080` to change the
defaults.

## Using the model directly

```python
import torch
from probabilistic_events.integrator import ProbabilisticEvents

model = ProbabilisticEvents(model_log_gabor=False).cuda()  # paper defaults otherwise
counts = torch.binomial(torch.full((256, 512, 8), 31.0), torch.full((256, 512, 8), 0.1))
flux, gradient, sample_weight, events = model(counts.cuda())  # (h, w, steps) counts
```

The input is photon counts per pixel, each summed over `sum_frames` binary
frames, with time as the last axis. State carries over between calls, so stream
the sequence in chunks. Flux and sample weight are updated every step; entropy
events compare the run-length entropy at the end of this call with the end of
the previous one, so they fire at the rate you call the model (once per output
frame in the CLI and dashboard).

## Speed

This release adds a fused Triton kernel for the run-length recursion. It keeps
each pixel's run-length state (and, for feature models, the per-hypothesis
feature means) in registers for the whole call, reuses the `lgamma` terms from
one step to the next, and computes the entropy and events in the same pass.
Features are still extracted by the PyTorch filters. The kernel is used
automatically on CUDA (`backend="auto"`); `backend="torch"` selects the
original `torch.compile` path.

The kernel implements the same recursion, with the same equations and update
order; the only differences are in float32 rounding (it drops constant terms
that cancel on normalisation and reuses `lgamma` values). `tests/test_triton_backend.py`
steps both backends from identical state and checks each against a float64
implementation; the kernel's error is never larger than the PyTorch path's.

Throughput in quanta frames per second (Q-FPS) of the Triton kernel on an
RTX 4090 (torch 2.6, CUDA 12.4): γ = 1e-6, K = 4, N_sum = 32 (Bernoulli:
N_sum = 1), synthetic Poisson counts, the whole module under `torch.compile`,
and the median of 190 timed calls. `uv run python scripts/benchmark_kernels.py`
reproduces it.

| Model | 256×256 | 512×512 | 1024×1024 |
|---|---:|---:|---:|
| Bernoulli (N_sum=1) | 27,509 | 27,712 | 10,334 |
| Binomial | 886,476 | 836,383 | 286,862 |
| Binomial + gradient (`fast`) | 573,806 | 209,380 | 55,739 |
| Binomial + DoG | 289,017 | 94,518 | 24,517 |
| Binomial + Log-Gabor (`default`) | 138,673 | 57,211 | 12,111 |

Log-Gabor uses 4 scales, 6 orientations and 12 PCA components here (the
model's default is 8 orientations and 24 components).

## Tests

```bash
uv run --group dev pytest tests/
```

The smoke tests skip themselves if `measuring_tape` hasn't been downloaded.

## Citation

```bibtex
@article{sundar2026probabilistic,
  title   = {Quanta Perception as Probabilistic Events},
  author  = {Sundar, Varun and Thodima, Pavan and Jungerman, Sacha and Gupta, Mohit},
  journal = {arXiv preprint arXiv:2608.27584},
  year    = {2026}
}
```
