"""
Fused Triton kernel for the binomial BOCPD loop of ProbabilisticEvents.

The PyTorch path launches a handful of kernels per time step and round-trips the
(h, w, memory) state through global memory each time. Here one program owns a
block of pixels, keeps their run-length state in registers, and loops over all
time steps of a forward call before writing the state back once.

Covers the binomial likelihood, the Gaussian feature likelihood (gradient,
Log-Gabor or DoG features, computed beforehand by the PyTorch filters) and their
combination, with max_filter_size == 1. Other configurations fall back to the
PyTorch implementation.
"""

import math

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _bocpd_binomial_kernel(
        photons_ptr,  # (T, N) float32, photon counts per summed frame
        logp_ptr,  # (N, K) log run-length distribution (unnormalised)
        start_ptr,  # (N, K) start index of each hypothesis, in binary frames
        alpha_ptr,  # (N, K) Beta posterior alpha
        beta_ptr,  # (N, K) Beta posterior beta
        recons_ptr,  # (N,) adaptive EMA reconstruction
        weight_out_ptr,  # (N,) sample weight of the last step
        insert_index_ptr,  # (N,) int64, slot chosen at the last step
        insert_mask_ptr,  # (N,) int8, whether the new hypothesis was inserted
        entropy_ptr,  # (N,) in: previous run-length entropy, out: current one
        events_ptr,  # (N,) int64 entropy-change event map (-1, 0, +1)
        feat_ptr,  # (T, C, N) float32 features (unused without HAS_FEAT)
        mu_ptr,  # (C, N, K) per-hypothesis feature means
        samples_ptr,  # (N, K) per-hypothesis sample counts
        N,
        T,
        t_abs0,  # binary frames processed before this call
        log_gamma,
        log_1m_gamma,
        alpha_0,
        beta_0,
        event_threshold,
        SUM_FRAMES: tl.constexpr,
        K: tl.constexpr,
        KP: tl.constexpr,  # K padded to a power of two
        BINOMIAL: tl.constexpr,  # binomial term in the likelihood
        COLLECT: tl.constexpr,  # track Beta statistics (binomial or Bayesian mean)
        HAS_FEAT: tl.constexpr,
        C: tl.constexpr,  # feature channels
        CP: tl.constexpr,  # C padded to a power of two
        USE_EMA: tl.constexpr,
        HAS_PREV_ENTROPY: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pix = pid * BLOCK + tl.arange(0, BLOCK)
        pix_ok = pix < N
        slot = tl.arange(0, KP)
        slot_ok = slot < K
        state_offs = pix[:, None] * K + slot[None, :]
        state_ok = pix_ok[:, None] & slot_ok[None, :]

        NEG_INF: tl.constexpr = float("-inf")
        POS_INF: tl.constexpr = float("inf")
        n = SUM_FRAMES * 1.0

        logp = tl.load(logp_ptr + state_offs, mask=state_ok, other=NEG_INF)
        start = tl.load(start_ptr + state_offs, mask=state_ok, other=0.0)
        alpha = tl.load(alpha_ptr + state_offs, mask=state_ok, other=1.0)
        beta = tl.load(beta_ptr + state_offs, mask=state_ok, other=1.0)
        recons = tl.load(recons_ptr + pix, mask=pix_ok, other=0.0)

        if HAS_FEAT:
            chan = tl.arange(0, CP)
            chan_ok = chan < C
            mu_offs = chan[None, :, None] * (N * K) + state_offs[:, None, :]
            mu_ok = chan_ok[None, :, None] & state_ok[:, None, :]
            mu = tl.load(mu_ptr + mu_offs, mask=mu_ok, other=0.0)
            samples = tl.load(samples_ptr + state_offs, mask=state_ok, other=1.0)
            feat_offs = chan[None, :] * N + pix[:, None]
            feat_ok = pix_ok[:, None] & chan_ok[None, :]
            log_2pi_c = C * 1.8378770664093453  # C * log(2 pi)

        if BINOMIAL and SUM_FRAMES > 1:
            # lgamma of the current statistics, carried across steps: this step's
            # lgamma(alpha + x) is next step's lgamma(alpha) (same float input)
            lg_a = libdevice.lgamma(alpha)
            lg_b = libdevice.lgamma(beta)
            lg_ab = libdevice.lgamma(alpha + beta)

        weight = tl.zeros([BLOCK], dtype=tl.float32)
        ins = tl.zeros([BLOCK], dtype=tl.int32)
        inserted = tl.zeros([BLOCK], dtype=tl.int1)

        for s in range(T):
            x = tl.load(photons_ptr + s * N + pix, mask=pix_ok, other=0.0)
            t_abs = (t_abs0 + (s + 1) * SUM_FRAMES) * 1.0
            occupied = tl.minimum(t_abs0 // SUM_FRAMES + s + 2, K)
            active = slot[None, :] < occupied

            # Sample weight: E_r[2 / (r + 2)] under the run-length posterior
            m = tl.max(logp, axis=1)
            prob = tl.exp(logp - m[:, None])
            prob = prob / tl.sum(prob, axis=1)[:, None]
            run_steps = tl.maximum(tl.maximum(t_abs - start, 1.0) / SUM_FRAMES, 1.0)
            weight = tl.sum(prob * (2.0 / (run_steps + 1.0)), axis=1)

            if USE_EMA:
                recons = (1.0 - weight) * recons + weight * (x / SUM_FRAMES)

            # Beta-binomial posterior predictive (constant log_binom term dropped:
            # it shifts every hypothesis equally and cancels on normalisation)
            xs = x[:, None]
            loglik = tl.zeros([BLOCK, KP], dtype=tl.float32)
            if not BINOMIAL:
                pass
            elif SUM_FRAMES == 1:
                loglik = tl.log(tl.where(xs > 0.5, alpha, beta)) - tl.log(alpha + beta)
            else:
                alpha_next = alpha + xs
                beta_next = beta + (n - xs)
                lg_a_next = libdevice.lgamma(alpha_next)
                lg_b_next = libdevice.lgamma(beta_next)
                lg_ab_next = libdevice.lgamma(alpha_next + beta_next)
                loglik = (lg_a_next - lg_a) + (lg_b_next - lg_b) - (lg_ab_next - lg_ab)

            if HAS_FEAT:
                # Gaussian predictive with unit variance per channel around the
                # running mean of each segment
                f = tl.load(feat_ptr + s * (C * N) + feat_offs, mask=feat_ok, other=0.0)
                fs = f[:, :, None]
                sq = tl.sum((fs - mu) * (fs - mu), axis=1)
                var = (samples + 1.0) / samples
                loglik -= 0.5 * (log_2pi_c + C * tl.log(var) + sq / var)

            # Run-length recursion
            joint = tl.where(active, logp + loglik, NEG_INF)
            jm = tl.max(joint, axis=1)
            new = log_gamma + jm + tl.log(tl.sum(tl.exp(joint - jm[:, None]), axis=1))
            logp = tl.where(active, logp + log_1m_gamma + loglik, logp)

            # Replace the weakest hypothesis if the new one beats it
            ins = tl.argmin(tl.where(slot_ok[None, :], logp, POS_INF), axis=1)
            is_ins = slot[None, :] == ins[:, None]
            beat = tl.sum(tl.where(is_ins, logp, 0.0), axis=1)
            inserted = new >= beat
            put = is_ins & inserted[:, None]
            logp = tl.where(put, new[:, None], logp)
            start = tl.where(put, t_abs, start)
            logp = logp - tl.max(logp, axis=1)[:, None]

            # Sufficient statistics (new segment includes the current frame)
            if COLLECT:
                alpha = tl.where(put, alpha_0, alpha) + xs
                beta = tl.where(put, beta_0, beta) + (n - xs)
            if HAS_FEAT:
                samples = tl.where(put, 0.0, samples) + 1.0
                mu = tl.where(put[:, None, :], 0.0, mu)
                mu += (fs - mu) / samples[:, None, :]
            if BINOMIAL and SUM_FRAMES > 1:
                # Reset slots restart from the prior, so their cache must be
                # recomputed; the rest reuse this step's lgamma values
                a_new = alpha_0 + x
                b_new = beta_0 + (n - x)
                lg_a = tl.where(put, libdevice.lgamma(a_new)[:, None], lg_a_next)
                lg_b = tl.where(put, libdevice.lgamma(b_new)[:, None], lg_b_next)
                lg_ab = tl.where(
                    put, libdevice.lgamma(a_new + b_new)[:, None], lg_ab_next
                )

        # Epilogue: entropy over active hypotheses and entropy-change events
        occupied = tl.minimum(t_abs0 // SUM_FRAMES + T + 1, K)
        active = slot[None, :] < occupied
        logits = tl.where(active, logp, NEG_INF)
        pa = tl.exp(logits - tl.max(logits, axis=1)[:, None])
        pa = pa / tl.sum(pa, axis=1)[:, None]
        pa = tl.where(active, pa, 0.0)
        entropy = -tl.sum(pa * tl.log(pa + 1e-10), axis=1)
        events = tl.zeros([BLOCK], dtype=tl.int64)
        if HAS_PREV_ENTROPY:
            prev = tl.load(entropy_ptr + pix, mask=pix_ok, other=0.0)
            delta = entropy - prev
            events = tl.where(delta > event_threshold, 1, events)
            events = tl.where(delta < -event_threshold, -1, events)
        if not USE_EMA and COLLECT:
            # Bayesian predictive mean, marginalised over all slots
            pm = tl.exp(logp - tl.max(logp, axis=1)[:, None])
            pm = pm / tl.sum(pm, axis=1)[:, None]
            recons = tl.sum(pm * (alpha / (alpha + beta + 1e-10)), axis=1)

        tl.store(entropy_ptr + pix, entropy, mask=pix_ok)
        tl.store(events_ptr + pix, events, mask=pix_ok)
        tl.store(logp_ptr + state_offs, logp, mask=state_ok)
        tl.store(start_ptr + state_offs, start, mask=state_ok)
        tl.store(alpha_ptr + state_offs, alpha, mask=state_ok)
        tl.store(beta_ptr + state_offs, beta, mask=state_ok)
        tl.store(recons_ptr + pix, recons, mask=pix_ok)
        if HAS_FEAT:
            tl.store(mu_ptr + mu_offs, mu, mask=mu_ok)
            tl.store(samples_ptr + state_offs, samples, mask=state_ok)
        tl.store(weight_out_ptr + pix, weight, mask=pix_ok)
        tl.store(insert_index_ptr + pix, ins.to(tl.int64), mask=pix_ok)
        tl.store(insert_mask_ptr + pix, inserted.to(tl.int8), mask=pix_ok)


def bocpd_binomial_triton(
    photon_cube: Tensor,
    logp: Tensor,
    start: Tensor,
    alpha: Tensor,
    beta: Tensor,
    recons: Tensor,
    t_abs0: int,
    gamma: float,
    sum_frames: int,
    alpha_0: float,
    beta_0: float,
    use_ema: bool,
    entropy: Tensor,
    has_prev_entropy: bool,
    event_threshold: float,
    binomial: bool = True,
    collect: bool = True,
    features: Tensor | None = None,
    mu: Tensor | None = None,
    samples: Tensor | None = None,
    block: int | None = None,
    num_warps: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Run the BOCPD loop over a (h, w, t) photon cube, updating the state tensors
    in place. With `features` (c, h, w, t), `mu` (c, h, w, K) and `samples`
    (h, w, K) the Gaussian feature likelihood is added.

    `entropy` holds the previous run-length entropy on entry and the current one
    on exit. With use_ema=False, `recons` is overwritten with the Bayesian
    predictive mean.

    :return: (sample_weight, insert_index, insert_mask, entropy_events).
    """
    h, w, steps = photon_cube.shape
    k = logp.shape[-1]
    n = h * w
    for state in (logp, start, alpha, beta, recons):
        assert state.is_contiguous() and state.dtype == torch.float32

    photons = photon_cube.float().reshape(n, steps).t().contiguous()
    weight = torch.empty((h, w), device=photon_cube.device, dtype=torch.float32)
    insert_index = torch.empty((h, w), device=photon_cube.device, dtype=torch.long)
    insert_mask = torch.empty((h, w), device=photon_cube.device, dtype=torch.int8)
    events = torch.empty((h, w), device=photon_cube.device, dtype=torch.long)

    has_feat = features is not None
    if has_feat:
        c = features.shape[0]
        for state in (mu, samples):
            assert state.is_contiguous() and state.dtype == torch.float32
        feats = features.float().reshape(c, n, steps).permute(2, 0, 1).contiguous()
    else:
        c = 1
        feats = mu = samples = logp  # unused placeholders

    # Tuned on an RTX 4090 across all variants and 256²-1024²: about two
    # threads per pixel when K pads to <= 4 slots, eight when it pads to 8
    kp = triton.next_power_of_2(k)
    if block is None:
        block = 64 if kp <= 4 else 32
    if num_warps is None:
        num_warps = 4 if kp <= 4 else 8

    grid = (triton.cdiv(n, block),)
    _bocpd_binomial_kernel[grid](
        photons,
        logp,
        start,
        alpha,
        beta,
        recons,
        weight,
        insert_index,
        insert_mask,
        entropy,
        events,
        feats,
        mu,
        samples,
        n,
        steps,
        t_abs0,
        math.log(gamma),
        math.log1p(-gamma),
        alpha_0,
        beta_0,
        event_threshold,
        SUM_FRAMES=sum_frames,
        K=k,
        KP=triton.next_power_of_2(k),
        BINOMIAL=binomial,
        COLLECT=collect,
        HAS_FEAT=has_feat,
        C=c,
        CP=triton.next_power_of_2(c),
        USE_EMA=use_ema,
        HAS_PREV_ENTROPY=has_prev_entropy,
        BLOCK=block,
        num_warps=num_warps,
    )
    return weight, insert_index, insert_mask.bool(), events
