"""Resolve a reference trajectory into the (w_k, M_k) pairs the selection
objective is defined over.

Both selection scripts optimise the same objective,

    D(alpha) = sum_k w_k || (1/m) sum_{i in I} g_i^k  -  gbar_k ||^2_{M_k},

    equivalently   P_ij = sum_k w_k * g_i^k^T M_k g_j^k,

and differ only in whether g is the classifier gradient (factored, in
``selection.kl``) or the gradient of every trainable parameter
(``selection.kl_full``). This module owns everything about (w_k, M_k)
so the two agree by construction.

Four objectives
───────────────
By default, each saved checkpoint receives the point weight
w_k = eta_k^p * a_k^2. This is the direct checkpoint approximation used by
the optimizer-aware scores. ``checkpoint_weighting='window'`` instead uses
sum_{s in window(k)} eta_s^p * a_s^2 when a checkpoint represents a window
of optimizer steps. M_k is either I or D_k^2, where
D_k = diag((sqrt(vhat_k) + eps)^-1).

--optimizer uniform     p = 0,  M_k = I
    Every checkpoint counts equally. Kept so old runs can be reproduced,
    not because it is right.

--optimizer sgd         p = 1,  M_k = I
    Two ULA chains that differ only in drift have per-step KL
    (beta*eta_s/4)*||grad Delta||^2, so checkpoint k carries eta_k by
    default. Window weighting integrates the learning rate over the steps
    represented by the checkpoint. Beta is a positive constant and cannot
    change which subset wins.

--optimizer adam        p = 1,  M_k = D_k^2
    The same KL weighting under preconditioned Langevin. The drift
    difference is eta_s * D_k * delta_g and the injected noise variance
    scales with eta_s, so the per-step KL is
    (eta_s^2 ||D_k delta_g||^2) / eta_s  =  eta_s ||delta_g||^2_{D_k^2}:
    the step size enters *linearly* and the preconditioner squared. This
    is the paper-inspired score.

--optimizer adam_update p = 2,  M_k = D_k^2
    Not a KL bound. This is the squared difference between the parameter
    updates themselves, eta_s^2 ||D_k delta_g||^2, i.e. how far the two
    runs' trajectories are pushed apart per step. A reasonable surrogate,
    and worth comparing against, but do not describe it as the
    manuscript's objective.

The momentum response a_s
─────────────────────────
a_s relates a gradient discrepancy delta_g to the discrepancy in the
bias-corrected first moment that Adam actually applies. It obeys

    delta_m_s = beta1 * delta_m_{s-1} + (1-beta1) * delta_g_s,
    delta_mhat_s = delta_m_s / (1 - beta1^s),

so the answer depends on how delta_g behaves across steps, and the two
limits differ by a lot rather than a little:

--momentum_response persistent (default)
    delta_g roughly constant across the window. Then
    delta_m_s = (1 - beta1^s) delta_g and delta_mhat_s = delta_g exactly,
    so a_s = 1. A subset's gradient discrepancy is a property of the
    subset, so it persists; this is the right default.

--momentum_response isolated
    delta_g appears at one step only, with both runs entering that step
    with the same moment. Then a_s = (1-beta1)/(1-beta1^s), which tends
    to 1-beta1 = 0.1 and so suppresses post-warmup steps by ~100x in
    a_s^2. Available for comparison; it models the wrong situation here.

Caveats that apply to every adam variant
────────────────────────────────────────
w_k >= 0 and M_k >= 0, so P stays positive semi-definite and every swap
identity the selectors rely on still holds — for this surrogate. It is a
frozen-preconditioner, local approximation: real subset training moves
both Adam moments, and the second moment in particular would make the
objective non-quadratic, at which point the exact swap formula no longer
applies. The manuscript proves no KL bound for deterministic Adam.

Checkpoint windows
──────────────────
Optional windows are in optimizer-step numbers and span the resolved trajectory,
so `--optimizer sgd/adam` actually sees the learning-rate schedule. The
default selection is 'spread'; 'prefix' reproduces the old behaviour of
taking the first K filenames, which for ``training.full_data``'s save
cadence means epochs 1-10 — a 3% slice of a 300-epoch run over which the
cosine schedule is flat, so any lr-aware weighting there is a no-op.

Weights are normalised to sum to 1, making D a weighted mean over the
resolved checkpoints rather than a sum. Scaling D by a positive constant
changes neither the swap rule nor the argmin, and it makes objective
values comparable across different --num_checkpoints.
"""

import glob
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from . import common

MANIFEST_NAME = 'trajectory.json'

# optimizer -> (power of eta_s in w_k, whether M_k = D_k^2)
SPEC = {
    'uniform':     (0, False),
    'sgd':         (1, False),
    'adam':        (1, True),
    'adam_update': (2, True),
}
OPTIMIZERS = tuple(SPEC)
MOMENTUM_RESPONSES = ('persistent', 'isolated')
METRICS = ('auto', 'none', 'adam')
CHECKPOINT_WEIGHTINGS = ('point', 'window')


def uses_metric(optimizer):
    """True when M_k = D_k^2, i.e. the checkpoint must carry vhat_k."""
    return SPEC[optimizer][1]


def resolve_metric(optimizer, override='auto'):
    """
    Whether M_k = D_k^2, letting the metric be chosen independently of
    the step weight.

    The two are separate modelling decisions and the coupling in SPEC is
    only a convenience. Keeping an override makes it possible to isolate
    checkpoint weighting from Adam's coordinate preconditioner, e.g.
    ``--optimizer uniform --metric adam`` versus
    ``--optimizer sgd --metric none``.
    """
    if override in (None, 'auto'):
        return uses_metric(optimizer)
    if override == 'none':
        return False
    if override == 'adam':
        return True
    raise ValueError(f"metric must be one of {METRICS}, got {override!r}")


def objective_tag(optimizer, metric='auto', checkpoint_weighting='point',
                  momentum='persistent'):
    """Stable filename suffix for objective variants that must not overwrite."""
    tag = '' if optimizer == 'uniform' else f'_{optimizer}'
    actual_metric = resolve_metric(optimizer, metric)
    if actual_metric != uses_metric(optimizer):
        tag += '_metric_adam' if actual_metric else '_metric_none'
    if checkpoint_weighting != 'point':
        tag += f'_{checkpoint_weighting}'
    if momentum != 'persistent':
        tag += f'_{momentum}'
    return tag


@dataclass
class Checkpoint:
    path: str
    step: int           # optimizer step s at which theta was snapshotted
    epoch: int
    lr: float           # eta_s
    weight: float       # w_k, normalised across the resolved set
    window: tuple       # (lo_step, hi_step) inclusive, the updates k stands for


# ─────────────────────────────────────────────────────────────────────────────
# Reading checkpoints written either way
#   training.reference writes a dict with the optimizer state alongside the
#   weights; training.full_data writes a bare state_dict. Everything here
#   accepts both, so an existing SGD trajectory stays usable and only the
#   adam objective actually requires a re-run.
# ─────────────────────────────────────────────────────────────────────────────
def load_record(path):
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'model' in obj:
        return obj
    return {'model': obj}


def load_weights(path):
    return load_record(path)['model']


def load_manifest(checkpoint_dir):
    path = os.path.join(checkpoint_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Step / epoch / learning-rate bookkeeping
# ─────────────────────────────────────────────────────────────────────────────
def steps_per_epoch(n_train, batch_size):
    """DataLoader default drop_last=False, so the tail batch counts."""
    return int(math.ceil(n_train / batch_size))


def _schedule_from_manifest(manifest):
    spe = int(manifest['steps_per_epoch'])
    epoch_lrs = np.asarray(manifest['epoch_lrs'], dtype=np.float64)
    return spe, epoch_lrs


def _schedule_reconstructed(n_train, batch_size, base_lr, num_epochs):
    """
    No manifest: rebuild the cosine schedule ``training.full_data`` used.
    CosineAnnealingLR is stepped once per epoch there, so eta is constant
    within an epoch and the per-epoch list determines every eta_s.
    """
    spe = steps_per_epoch(n_train, batch_size)
    epoch_lrs = np.array([common.cosine_lr(base_lr, e, num_epochs)
                          for e in range(1, num_epochs + 1)], dtype=np.float64)
    return spe, epoch_lrs


def _step_lrs(spe, epoch_lrs):
    """eta_s for s = 1 .. total_steps, as a 1-indexed-friendly array."""
    return np.repeat(epoch_lrs, spe)


# ─────────────────────────────────────────────────────────────────────────────
# Selection and windows
# ─────────────────────────────────────────────────────────────────────────────
def _available(checkpoint_dir, manifest):
    """[(path, step, epoch, lr)], sorted by step."""
    if manifest is not None:
        out = [(os.path.join(checkpoint_dir, c['file']), int(c['step']),
                int(c['epoch']), float(c['lr'])) for c in manifest['checkpoints']]
        return sorted(out, key=lambda r: r[1])

    paths = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pth')))
    if not paths:
        raise ValueError(f"No checkpoint_*.pth in {checkpoint_dir}")
    return [(p, 0, common.checkpoint_epoch(p), 0.0) for p in paths]


def _pick(available, num_checkpoints, selection, epochs, steps):
    if epochs:
        by_epoch = {r[2]: r for r in available}
        missing = [e for e in epochs if e not in by_epoch]
        if missing:
            raise ValueError(f"No checkpoint at epoch(s) {missing}. "
                             f"Available: {sorted(by_epoch)}")
        return [by_epoch[e] for e in sorted(epochs)]

    if steps:
        by_step = {r[1]: r for r in available}
        missing = [s for s in steps if s not in by_step]
        if missing:
            raise ValueError(f"No checkpoint at step(s) {missing}. "
                             f"Available: {sorted(by_step)}")
        return [by_step[s] for s in sorted(steps)]

    if num_checkpoints <= 0 or num_checkpoints >= len(available):
        return list(available)

    if selection == 'prefix':
        return list(available[:num_checkpoints])
    if selection != 'spread':
        raise ValueError(f"Unknown selection: {selection!r}")

    # Evenly spaced by position, endpoints included, so the resolved set
    # actually spans the run instead of clustering at its start.
    pos = np.unique(np.linspace(0, len(available) - 1, num_checkpoints).round().astype(int))
    return [available[i] for i in pos]


def _windows(steps, total_steps):
    """
    Midpoint windows in step numbers. The last window ends at the last
    resolved checkpoint, not at total_steps: a truncated trajectory should
    weight the part of training it observed and claim nothing past it.
    """
    bounds = []
    for t, s in enumerate(steps):
        lo = 1 if t == 0 else math.floor((steps[t - 1] + s) / 2) + 1
        hi = s if t == len(steps) - 1 else math.floor((s + steps[t + 1]) / 2)
        bounds.append((min(lo, s), max(lo, min(hi, total_steps))))
    return bounds


def momentum_response(steps, beta1, momentum):
    """a_s, the factor relating a gradient discrepancy to the applied one."""
    if momentum == 'persistent':
        return np.ones_like(steps, dtype=np.float64)
    if momentum == 'isolated':
        return (1.0 - beta1) / (1.0 - np.power(beta1, steps))
    raise ValueError(f"momentum_response must be one of {MOMENTUM_RESPONSES}, "
                     f"got {momentum!r}")


def _weights(optimizer, windows, step_lrs, beta1, momentum='persistent',
             checkpoint_steps=None, checkpoint_lrs=None,
             checkpoint_weighting='point'):
    if optimizer not in SPEC:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")
    if checkpoint_weighting not in CHECKPOINT_WEIGHTINGS:
        raise ValueError(
            f"checkpoint_weighting must be one of {CHECKPOINT_WEIGHTINGS}, "
            f"got {checkpoint_weighting!r}"
        )
    power, metric = SPEC[optimizer]

    if power == 0:
        w = np.ones(len(windows), dtype=np.float64)
    elif checkpoint_weighting == 'point':
        if checkpoint_steps is None or checkpoint_lrs is None:
            raise ValueError("Point weighting requires checkpoint steps and learning rates.")
        checkpoint_steps = np.asarray(checkpoint_steps, dtype=np.float64)
        checkpoint_lrs = np.asarray(checkpoint_lrs, dtype=np.float64)
        w = checkpoint_lrs ** power
        if metric:
            w = w * momentum_response(
                checkpoint_steps, beta1, momentum) ** 2
    else:
        per_step = step_lrs ** power
        if metric:
            s = np.arange(1, len(step_lrs) + 1, dtype=np.float64)
            per_step = per_step * momentum_response(s, beta1, momentum) ** 2
        cum = np.concatenate([[0.0], np.cumsum(per_step)])
        w = np.array([cum[min(hi, len(per_step))] - cum[max(lo - 1, 0)]
                      for lo, hi in windows], dtype=np.float64)

    if not np.all(np.isfinite(w)) or w.sum() <= 0:
        raise ValueError(f"Degenerate checkpoint weights for optimizer={optimizer!r}: {w}")
    return w / w.sum()


def resolve(checkpoint_dir, *, optimizer='uniform', num_checkpoints=10,
            selection='spread', epochs=None, steps=None,
            n_train=None, batch_size=128, base_lr=0.01, num_epochs=None,
            beta1=0.9, momentum='persistent', metric='auto',
            checkpoint_weighting='point'):
    """
    -> (checkpoints, manifest, description).

    `manifest` is None for a legacy ``training.full_data`` directory, in which
    case eta_s is reconstructed from the cosine schedule and step numbers
    are derived from epoch x steps_per_epoch. That is exact for SGD, and is
    why uniform/sgd need no re-run; the adam variants do, because vhat_k is
    not recoverable from weights alone.
    """
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {optimizer!r}")
    needs_vhat = resolve_metric(optimizer, metric)

    manifest = load_manifest(checkpoint_dir)
    if manifest is not None:
        spe, epoch_lrs = _schedule_from_manifest(manifest)
        ref_optimizer = manifest.get('optimizer', 'sgd')
        num_epochs = int(manifest['num_epochs'])
        beta1 = float(manifest.get('betas', (beta1, 0.999))[0])
    else:
        if needs_vhat:
            asked = (f"--metric {metric}" if metric not in (None, 'auto')
                     else f"--optimizer {optimizer}")
            raise ValueError(
                f"{asked} sets M_k = D_k^2, which needs the second moments vhat_k; a "
                f"bare state_dict does not contain them. {checkpoint_dir} has no "
                f"{MANIFEST_NAME}, so it was written by an older full-data trainer "
                f"(weights only). Train an Adam reference with "
                f"scripts/train/reference.py --optimizer "
                f"adam first, or pass --metric none."
            )
        if n_train is None or num_epochs is None:
            raise ValueError("Legacy checkpoint dir: n_train and num_epochs are required "
                             "to reconstruct the learning-rate schedule.")
        spe, epoch_lrs = _schedule_reconstructed(n_train, batch_size, base_lr, num_epochs)
        ref_optimizer = 'sgd'

    if needs_vhat and ref_optimizer not in ('adam', 'adamw'):
        raise ValueError(
            f"M_k = D_k^2 was requested (--optimizer {optimizer} --metric {metric}) but "
            f"the reference trajectory in {checkpoint_dir} was trained with "
            f"{ref_optimizer!r}. Scoring an SGD trajectory with an Adam metric mixes two "
            f"different training runs."
        )

    step_lrs = _step_lrs(spe, epoch_lrs)
    total_steps = len(step_lrs)

    available = _available(checkpoint_dir, manifest)
    if manifest is None:
        available = [(p, e * spe, e, float(epoch_lrs[min(e, num_epochs) - 1]))
                     for p, _, e, _ in available]

    picked = _pick(available, num_checkpoints, selection, epochs, steps)
    picked_steps = [r[1] for r in picked]
    picked_lrs = [r[3] for r in picked]
    windows = _windows(picked_steps, total_steps)
    weights = _weights(
        optimizer, windows, step_lrs, beta1, momentum,
        checkpoint_steps=picked_steps, checkpoint_lrs=picked_lrs,
        checkpoint_weighting=checkpoint_weighting)

    checkpoints = [Checkpoint(path=p, step=s, epoch=e, lr=lr, weight=float(w), window=win)
                   for (p, s, e, lr), w, win in zip(picked, weights, windows)]

    power = SPEC[optimizer][0]
    lo, hi = windows[0][0], windows[-1][1]
    frac_steps = (hi - lo + 1) / total_steps
    frac_eta = float(step_lrs[lo - 1:hi].sum() / step_lrs.sum())
    span = f"epochs {checkpoints[0].epoch}-{checkpoints[-1].epoch}"
    if power == 0:
        weight_formula = "1"
    elif checkpoint_weighting == 'point':
        weight_formula = f"eta_k^{power}"
    else:
        weight_formula = f"sum_window eta_s^{power}"
    description = (f"{len(checkpoints)} checkpoints ({span}, "
                   f"steps {checkpoints[0].step}-{checkpoints[-1].step} of {total_steps}), "
                   f"objective={optimizer} "
                   f"(w_k = {weight_formula}"
                   + (f" a_s^2 [{momentum}]" if needs_vhat else "")
                   + f", M_k = {'D_k^2' if needs_vhat else 'I'}), selection="
                   f"{'explicit' if (epochs or steps) else selection}, "
                   f"checkpoint_weighting={checkpoint_weighting}, "
                   f"w_max/w_min={weights.max() / weights.min():.1f}, "
                   f"covers {frac_steps:.1%} of steps / {frac_eta:.1%} of sum eta")

    # A short prefix trajectory sits on a flat stretch of the cosine
    # schedule, where sum eta_s over a window is just the window width in
    # disguise. Midpoint windows are lopsided (the first reaches back to
    # step 1, the last stops at its own checkpoint), so the resulting
    # weights look principled and are not. Say so rather than let the
    # w_k column imply otherwise.
    if power > 0 and checkpoint_weighting == 'window':
        eta_lo = float(step_lrs[lo - 1:hi].min())
        eta_hi = float(step_lrs[lo - 1:hi].max())
        if eta_hi <= eta_lo * 1.01:
            description += (
                f". NOTE eta varies by only {100 * (eta_hi / eta_lo - 1):.2f}% over this "
                f"span, so w_k carries no learning-rate information and reduces to the "
                f"window widths ({weights.max() / weights.min():.1f}:1 first to last). "
                f"For equal weights use --optimizer uniform"
                + (" --metric adam" if needs_vhat else ""))
    return checkpoints, manifest, description


def describe(checkpoints):
    lines = [f"{'epoch':>7}{'step':>9}{'lr':>12}{'window':>18}{'w_k':>10}"]
    for c in checkpoints:
        lines.append(f"{c.epoch:>7}{c.step:>9}{c.lr:>12.6f}"
                     f"{f'{c.window[0]}-{c.window[1]}':>18}{c.weight:>10.5f}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# M_k
#   vhat is bias-corrected with the step actually stored in the record, so
#   early checkpoints are handled correctly rather than being dominated by
#   the 1 - beta2^s factor.
# ─────────────────────────────────────────────────────────────────────────────
def _vhat(record):
    if 'exp_avg_sq' not in record or record['exp_avg_sq'] is None:
        raise ValueError(
            f"Checkpoint has no 'exp_avg_sq'; it was not written by "
            f"scripts/train/reference.py --optimizer adam."
        )
    v = record['exp_avg_sq']
    step = float(record.get('step', 0))
    beta2 = float(record.get('betas', (0.9, 0.999))[1])
    correction = (1.0 - beta2 ** step) if step > 0 else 1.0
    return {name: (t.double() / correction) for name, t in v.items()}


def adam_metric_flat(checkpoint: Checkpoint, named_params, eps=1e-8, power=2):
    """
    M_k = (1/(sqrt(vhat) + eps))^power as one flat vector, laid out in the
    order of `named_params` — the same order ``objective.flat_grads`` uses.
    power=2 is D_k^2 (the update-matching metric); power=1 is D_k.
    """
    vhat = _vhat(load_record(checkpoint.path))
    flat = []
    for name, p in named_params:
        if name not in vhat:
            raise ValueError(f"No Adam second moment saved for parameter {name!r}.")
        v = vhat[name]
        if tuple(v.shape) != tuple(p.shape):
            raise ValueError(f"vhat[{name!r}] has shape {tuple(v.shape)}, "
                             f"parameter has {tuple(p.shape)}.")
        flat.append(v.reshape(-1))
    return (1.0 / (torch.cat(flat).sqrt() + eps)) ** power


def adam_metric_factored(checkpoint: Checkpoint, num_classes, feat_dim,
                         eps=1e-8, power=2, include_bias=True,
                         weight_name='linear.weight', bias_name='linear.bias'):
    """
    M_k for the classifier gradient g_i = L_i (x) Phi_i, shaped
    [num_classes, feat_dim(+1)]. The trailing column is linear.bias, which
    pairs with the constant 1 appended to each Phi_i: the bias gradient is
    exactly the logit gradient, so <g_i, g_j> picks up an extra
    <L_i, L_j> * 1 term.
    """
    vhat = _vhat(load_record(checkpoint.path))
    if weight_name not in vhat:
        raise ValueError(f"No Adam second moment saved for {weight_name!r}.")
    v = vhat[weight_name]
    if tuple(v.shape) != (num_classes, feat_dim):
        raise ValueError(f"vhat[{weight_name!r}] has shape {tuple(v.shape)}, "
                         f"expected {(num_classes, feat_dim)}.")
    if include_bias:
        if bias_name not in vhat:
            raise ValueError(f"--include_bias but no second moment for {bias_name!r}.")
        v = torch.cat([v, vhat[bias_name].reshape(num_classes, 1)], dim=1)
    return (1.0 / (v.sqrt() + eps)) ** power


# Re-exported so callers that already import this module do not need to
# reach into common for it; it lives there because
# common.retrain_on_subset needs it and cannot import this module.
build_optimizer = common.build_optimizer
