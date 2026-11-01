"""
The fused Triton kernel must implement exactly the same recursion as the PyTorch
path. Both run in float32 with different op orders, and the Beta-binomial term is
a difference of large lgamma values, so they round differently; near-ties in the
"replace the weakest hypothesis" test can then go either way.

To check the kernel is exact up to rounding (and no less accurate than the
PyTorch path), we step both from identical state and compare each against a
float64 implementation of the same recursion.
"""

import math

import pytest
import torch

from probabilistic_events.integrator import ProbabilisticEvents
from probabilistic_events.integrator.triton_kernels import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()), reason="needs CUDA + Triton"
)

STATE = (
    "log_forecaster_distribution_ll",
    "forecaster_distribution_index_ll",
    "alpha_ll",
    "beta_ll",
    "recons",
    "prev_forecaster_entropy",
)


def _stream(h, w, steps, sum_frames, seed=0):
    """Piecewise-constant flux with a step change halfway through."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    p0 = torch.rand(h, w, 1, device="cuda", generator=g) * 0.5
    p1 = torch.rand(h, w, 1, device="cuda", generator=g) * 0.5
    p = torch.cat(
        [p0.expand(h, w, steps // 2), p1.expand(h, w, steps - steps // 2)], -1
    ).contiguous()
    return torch.binomial(torch.full_like(p, float(sum_frames)), p, generator=g)


def _model(backend, sum_frames, memory_size, bayesian=False):
    return ProbabilisticEvents(
        bocpd_gamma=1e-3,
        sum_frames=sum_frames,
        memory_size=memory_size,
        model_binomial=True,
        model_log_gabor=False,
        use_bayesian_predictive=bayesian,
        backend=backend,
    ).cuda()


def _reference_step(m, x):
    """One step of the binomial recursion in float64, from m's current state."""
    sf, k = m.sum_frames, m.memory_size
    logp = m.log_forecaster_distribution_ll.double()
    start = m.forecaster_distribution_index_ll.double()
    a, b = m.alpha_ll.double(), m.beta_ll.double()
    x = x.double().unsqueeze(-1)
    t_abs = m.t_absolute + sf
    occupied = min(t_abs // sf + 1, k)
    active = torch.arange(k, device=x.device) < occupied

    def lbeta(p, q):
        return torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)

    loglik = lbeta(a + x, b + sf - x) - lbeta(a, b)
    joint = (logp + loglik).masked_fill(~active, -math.inf)
    new = math.log(m.bocpd_gamma) + torch.logsumexp(joint, -1)
    logp = torch.where(active, logp + math.log1p(-m.bocpd_gamma) + loglik, logp)
    gap = new - logp.min(-1).values
    return torch.softmax(logp, -1), new, logp, gap


@pytest.mark.parametrize("sum_frames,memory_size", [(1, 5), (32, 5), (8, 8), (4, 3)])
def test_triton_as_accurate_as_torch(sum_frames, memory_size):
    cube = _stream(48, 40, 210, sum_frames)
    ref = _model("torch", sum_frames, memory_size)
    tri = _model("triton", sum_frames, memory_size)

    flips = {"torch": 0, "triton": 0}
    err = {"torch": 0.0, "triton": 0.0}
    max_tie_gap = 0.0
    for t in range(cube.shape[-1]):
        x = cube[..., t : t + 1].contiguous()
        if ref.t_absolute > 0:
            _, new64, logp64, gap64 = _reference_step(ref, x[..., 0])
            want_insert = gap64 >= 0
        ref(x)
        tri(x)
        assert ref.t_absolute == tri.t_absolute
        if t == 0:
            continue

        for name, m in (("torch", ref), ("triton", tri)):
            wrong = m.insert_mask != want_insert
            flips[name] += wrong.sum().item()
            if wrong.any():
                # Only genuine near-ties may be decided differently from float64
                max_tie_gap = max(max_tie_gap, gap64[wrong].abs().max().item())
            # Run-length posterior error on pixels with the float64 decision
            ok = ~wrong
            p = m.forecaster_distribution_ll.double()
            ins = m.insert_index.unsqueeze(-1)
            put = want_insert.unsqueeze(-1)
            logp_after = logp64.scatter(
                -1, ins, torch.where(put, new64.unsqueeze(-1), logp64.gather(-1, ins))
            )
            p64 = torch.softmax(logp_after, -1)
            err[name] = max(err[name], (p - p64)[ok].abs().max().item())

        for name in STATE:
            getattr(tri, name).copy_(getattr(ref, name))

    total = (cube.shape[-1] - 1) * cube.shape[0] * cube.shape[1]
    print(f"\nsum_frames={sum_frames} flips={flips} / {total} err={err} max_tie_gap={max_tie_gap:.2e}")
    assert max_tie_gap < 1e-2
    assert flips["triton"] <= max(2 * flips["torch"], 5)
    assert err["triton"] <= max(2 * err["torch"], 1e-5)


@pytest.mark.parametrize("sum_frames", [1, 32])
@pytest.mark.parametrize("bayesian", [False, True])
def test_triton_end_to_end_close(sum_frames, bayesian):
    """Without resyncing, tie flips stay rare and outputs agree."""
    cube = _stream(64, 64, 400, sum_frames, seed=1)
    ref = _model("torch", sum_frames, 5, bayesian)
    tri = _model("triton", sum_frames, 5, bayesian)
    for x in cube.split(25, dim=-1):
        out_ref, out_tri = ref(x.contiguous()), tri(x.contiguous())
    for a, b in (out_ref[0], out_tri[0]), (out_ref[2], out_tri[2]):
        diff = (a.float() - b.float()).abs()
        assert diff.mean() < 5e-4
        assert (diff > 1e-3).float().mean() < 0.02
    assert (out_ref[3] != out_tri[3]).float().mean() < 0.01


@pytest.mark.parametrize("sum_frames", [1, 32])
def test_triton_chunking_invariant(sum_frames):
    """The lgamma cache carried inside a call must equal recomputing per call."""
    cube = _stream(64, 64, 120, sum_frames, seed=2)
    one = _model("triton", sum_frames, 5)
    many = _model("triton", sum_frames, 5)
    for x in cube.split(1, dim=-1):
        one(x.contiguous())
    for x in cube.split(40, dim=-1):
        many(x.contiguous())
    for name in STATE[:5]:
        assert torch.equal(getattr(one, name), getattr(many, name)), name


@pytest.mark.parametrize(
    "features",
    [
        dict(model_gradient=True, model_log_gabor=False),
        dict(model_log_gabor=True),
        dict(model_log_gabor=True, model_binomial=False),
        dict(model_log_gabor=True, use_bayesian_predictive=True),
    ],
)
def test_triton_feature_models_close(features):
    """Feature likelihoods (features from the PyTorch filters) on the fused kernel."""
    cube = _stream(64, 64, 120, 31, seed=3)
    outs = []
    for backend in ("torch", "triton"):
        m = ProbabilisticEvents(bocpd_gamma=1e-3, backend=backend, **features).cuda()
        for x in cube.split(20, dim=-1):
            out = m(x.contiguous())
        outs.append((out, m))
    (ref, m_ref), (tri, m_tri) = outs
    for a, b in (ref[0], tri[0]), (ref[2], tri[2]):
        diff = (a.float() - b.float()).abs()
        assert diff.mean() < 5e-4
        assert (diff > 1e-3).float().mean() < 0.02
    assert (ref[3] != tri[3]).float().mean() < 0.01
    if not m_ref.model_binomial:
        # No lgamma terms, so no near-tie flips: the state must match closely
        torch.testing.assert_close(m_ref.feature_mu_ll, m_tri.feature_mu_ll)
        torch.testing.assert_close(m_ref.samples_ll, m_tri.samples_ll)
