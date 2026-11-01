"""
Runtime of ProbabilisticEvents on synthetic photon cubes (no I/O), in quanta
frames per second (Q-FPS), for PyTorch (torch.compile) and the fused Triton kernel.

Follows the paper's runtime protocol (scripts/runtime.py with
conf/visualization_runtime.yaml in the paper's experiment code), so the PyTorch
column is directly comparable with the arXiv version:

- gamma = 1e-6, K = 4; Bernoulli uses N_sum = 1 with 8 binary frames per call,
  every other variant N_sum = 32 with 256 binary frames per call
- Poisson photon counts with mean 0.08 * N_sum (clipped to N_sum)
- the whole module wrapped in torch.compile, one untimed setup call
- 200 timed calls (CUDA events), the first 10 discarded, median reported

Usage:
    uv run python scripts/benchmark_kernels.py [--backends torch triton] [--models ...]
"""

import argparse
import platform
from statistics import median

import torch
from tabulate import tabulate

from probabilistic_events.integrator import ProbabilisticEvents

torch.set_float32_matmul_precision("high")

GAMMA = 1e-6
MEMORY_SIZE = 4
ITERATIONS = 200
DISCARD_FIRST = 10

FEATURES_OFF = dict(model_gradient=False, model_log_gabor=False, model_difference_of_gaussians=False)
MODELS = {
    # name: (sum_frames, binary frames per call, ProbabilisticEvents kwargs)
    "Bernoulli": (1, 8, FEATURES_OFF),
    "Binomial": (32, 256, FEATURES_OFF),
    "Gradient": (32, 256, {**FEATURES_OFF, "model_gradient": True}),
    "Difference of Gaussians": (32, 256, {**FEATURES_OFF, "model_difference_of_gaussians": True, "num_scales": 4}),
    # 6 orientations, as in the run behind the paper's runtime figure (the class default is 8)
    "Log-Gabor": (32, 256, {**FEATURES_OFF, "model_log_gabor": True, "num_orientations": 6, "pca_components": 12}),
}
RESOLUTIONS = [(256, 256), (512, 512), (1024, 1024)]


def make_photon_cube(h: int, w: int, steps: int, sum_frames: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(1234 + h * 17 + w * 31 + steps)
    rate = torch.full((h, w, steps), max(0.08 * sum_frames, 1.0), device="cuda")
    return torch.poisson(rate, generator=generator).clamp_(0, sum_frames).contiguous()


@torch.no_grad()
def benchmark(backend: str, name: str, h: int, w: int) -> float:
    sum_frames, quanta_per_call, kwargs = MODELS[name]
    model = ProbabilisticEvents(
        bocpd_gamma=GAMMA,
        memory_size=MEMORY_SIZE,
        sum_frames=sum_frames,
        model_binomial=True,
        backend=backend,
        **kwargs,
    ).cuda()
    torch.compiler.reset()
    model = torch.compile(model)
    cube = make_photon_cube(h, w, quanta_per_call // sum_frames, sum_frames)

    model(cube)  # setup: compilation, state allocation, filter subspace fit
    torch.cuda.synchronize()
    seconds = []
    for _ in range(ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(cube)
        end.record()
        torch.cuda.synchronize()
        seconds.append(start.elapsed_time(end) / 1e3)
    del model
    torch.cuda.empty_cache()
    return quanta_per_call / median(seconds[DISCARD_FIRST:])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backends", nargs="+", default=["torch", "triton"], choices=["torch", "triton"])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--max-resolution", type=int, default=1024)
    args = parser.parse_args()

    rows = []
    for name in args.models:
        for h, w in RESOLUTIONS:
            if max(h, w) > args.max_resolution:
                continue
            qfps = {}
            for backend in args.backends:
                try:
                    qfps[backend] = benchmark(backend, name, h, w)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
            row = [name, f"{h}x{w}"] + [f"{qfps[b]:,.0f}" if b in qfps else "OOM" for b in args.backends]
            if {"torch", "triton"} <= qfps.keys():
                row.append(f"{qfps['triton'] / qfps['torch']:.1f}x")
            rows.append(row)
            print(tabulate([row], tablefmt="plain"), flush=True)

    headers = ["Model", "Resolution"] + [f"{b} Q-FPS" for b in args.backends]
    if {"torch", "triton"} <= set(args.backends):
        headers.append("Speedup")
    print()
    print(f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__} | Python {platform.python_version()}")
    print(tabulate(rows, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
