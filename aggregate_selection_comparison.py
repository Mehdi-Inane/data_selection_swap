#!/usr/bin/env python3
"""
aggregate_selection_comparison.py

Summarises KL-Faithful vs. every baseline in the selection suite on three
quantities of interest, mean ± std over seeds, per (fraction, method, K):

  * final-epoch test accuracy                         (<prefix>_metrics.json)
  * selection time, three ways                        (<prefix>_timing.json)
      selection  : wall-clock of the method's own work (extra training +
                   scoring + solver) — what the method adds on top of the
                   shared full-data trajectory
      reference  : time of the first `reference_epochs_used` epochs of the
                   shared trajectory (train_full_data.py's train_time.json),
                   i.e. the part of it the method actually needed
      total      : selection + reference — the end-to-end cost of producing
                   the subset from scratch
  * D_alpha                                            (<prefix>_indices.pt)
      the KL-Faithful optimization objective, evaluated at the method's own
      selected subset:

          D_alpha = (1/m^2) * alpha^T P0 alpha + alpha^T q

      with alpha the {0,1}^n indicator of the selected indices, and

          (P0)_ij = sum_k <grad l(theta_k, Z_i), grad l(theta_k, Z_j)>
          q_i     = -(2/m) * sum_k <grad l(theta_k, Z_i), grad L_S(theta_k)>

      summed over the checkpoints theta_k of the reference trajectory. This
      is computed for EVERY method's indices, not just kl_faithful's — it
      answers "how good is method X's subset according to our own
      objective?", which is a like-for-like comparison since every method's
      alpha is scored against the identical trajectory kernel.

Checkpoint-count (K) awareness
───────────────────────────────
train_kl_selection.py (and any other trajectory-based method, e.g. MoSo,
that adopts the same convention) tags every saved file with
`_K<num_checkpoints_used>`, reflecting how many checkpoints were actually
used to build the trajectory kernel for that run — the *effective* count
after clamping to what --checkpoint_dir actually contains, not just the
requested --num_checkpoints value. A single run_dir can therefore hold
several (metrics, timing, indices) file triples for the same (dataset,
budget, seed) — one per K that was swept.

Rather than assuming a fixed filename, this script globs each run_dir for
every "<method>_<budget>[_K<k>]_metrics.json" it finds, pairs each one with
its matching "..._timing.json" and "..._indices.pt" by shared filename
prefix, and groups results by (fraction, method, K) instead of just
(fraction, method). Methods/files that don't tag a checkpoint count (every
non-trajectory baseline: herding, moderate coreset, graphcut, facility
location, EL2N, random) fall back to a single row with K shown as '–'. Use
--checkpoints to restrict the report to specific K values.

D_alpha scoring trajectory
────────────────────────────
Computing D_alpha requires rebuilding the (L, Phi) trajectory features —
the same expensive extraction train_kl_selection.py itself performs — so
it needs --data_dir and --checkpoint_dir and is OFF by default (the plain
accuracy/timing table never needed a GPU or the dataset; this keeps that
fast path unchanged). Pass --data_dir to turn it on. Two rules decide
which trajectory scores a given row:
  * A row with its own K tag (kl_faithful, or any method following the
    same convention) is scored on ITS OWN K-checkpoint trajectory — the
    exact one it was optimized against. For kl_faithful this is a built-in
    sanity check: the recomputed value should match (up to floating point)
    the final entry of that run's own "<prefix>_objective.json".
  * A row with no K tag (every one-shot baseline) is scored on a single
    reference trajectory of length --baseline_score_checkpoints (default
    10), since those methods don't define their own K.
Trajectories are cached in memory per (seed, K) within one run of this
script, since the same (seed, K) trajectory is reused across every
fraction and every K-tagged method that shares it.

Usage:
    python aggregate_selection_comparison.py --dataset cifar100
    python aggregate_selection_comparison.py --dataset cifar100 --checkpoints 10
    python aggregate_selection_comparison.py --dataset cifar100 \
        --data_dir $SLURM_TMPDIR/cifar100_data   # turns on the D_alpha column
"""

import argparse
import csv
import json
import os
import re

import numpy as np

N_TRAIN = {'cifar100': 45000, 'imagenet': 1281167 - int(0.1 * 1281167)}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def run_dir(scratch, method, dataset, budget, seed):
    if method == 'kl_faithful':
        return os.path.join(scratch, 'gradmatch_swap', dataset, str(budget), f'seed_{seed}')
    return os.path.join(scratch, 'data_selection_baselines', method, dataset,
                        str(budget), f'seed_{seed}')


def discover_methods(scratch, dataset):
    root = os.path.join(scratch, 'data_selection_baselines')
    found = sorted(m for m in os.listdir(root)
                   if os.path.isdir(os.path.join(root, m, dataset))) if os.path.isdir(root) else []
    return ['kl_faithful'] + found


def _metrics_pattern(method, budget):
    """Matches '<method>_<budget>_metrics.json' or '<method>_<budget>_K<k>_metrics.json'."""
    return re.compile(rf'^{re.escape(method)}_{budget}(?:_K(\d+))?_metrics\.json$')


def discover_runs(d, method, budget):
    """
    Find every (K, metrics_path, timing_path) triple in run_dir `d` for this
    method/budget, by globbing rather than assuming a single fixed filename.
    K is None for files that don't tag a checkpoint count. metrics_path
    always exists (it's the glob source); timing_path is paired by the
    shared filename prefix and is None if no matching timing file exists.

    Signature intentionally unchanged (still a 3-tuple) so existing callers
    (e.g. plot_checkpoint_ablation.py) keep working unmodified. Use
    indices_path_for(metrics_path) to get the matching indices file.
    """
    if not os.path.isdir(d):
        return []
    pattern = _metrics_pattern(method, budget)
    runs = []
    for fname in sorted(os.listdir(d)):
        m = pattern.match(fname)
        if not m:
            continue
        k = int(m.group(1)) if m.group(1) is not None else None
        prefix = fname[:-len('_metrics.json')]
        timing_path = os.path.join(d, f'{prefix}_timing.json')
        runs.append((k, os.path.join(d, fname),
                     timing_path if os.path.exists(timing_path) else None))
    return runs


def indices_path_for(metrics_path):
    """'<dir>/<prefix>_metrics.json' -> '<dir>/<prefix>_indices.pt' if it exists, else None."""
    d = os.path.dirname(metrics_path)
    fname = os.path.basename(metrics_path)
    prefix = fname[:-len('_metrics.json')]
    p = os.path.join(d, f'{prefix}_indices.pt')
    return p if os.path.exists(p) else None


def mean_std(values):
    return (float(np.mean(values)), float(np.std(values))) if values else (float('nan'),) * 2


# ─────────────────────────────────────────────────────────────────────────────
# D_alpha scoring
#   Only imports torch / torchvision / cords (via selection_common and
#   train_kl_selection) on first use, so the plain accuracy/timing table
#   (the default) never needs them.
# ─────────────────────────────────────────────────────────────────────────────
class ObjectiveScorer:
    """
    Rebuilds and caches the (LP_list, c_sum) trajectory kernel for each
    (seed, K) pair actually needed, then evaluates D_alpha for any method's
    saved indices.pt against that cached trajectory. LP_list/c_sum for a
    given (seed, K) is reused across every fraction and every K-tagged
    method that shares it, since neither depends on the fraction/method —
    only on the seed (it fixes the train/val split) and K (how many
    checkpoints feed the kernel).
    """

    def __init__(self, dataset, data_dir, checkpoint_dir, download,
                batch_size, num_workers, proj_dim):
        import torch
        import selection_common as sc
        import train_kl_selection as kl

        self.torch = torch
        self.sc = sc
        self.kl = kl
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.proj_dim = proj_dim

        self.cfg = self._load_dataset_config(sc, dataset, data_dir, download)
        self.ckpt_paths_all = sc.list_checkpoints(checkpoint_dir)

        self._loader_cache = {}   # seed -> (eval_loader, n_train)
        self._traj_cache   = {}   # (seed, K) -> (LP_list, c_sum, n_train)
        self._model = None

    @staticmethod
    def _load_dataset_config(sc, dataset, data_dir, download):
        try:
            return sc.get_dataset_config(dataset, data_dir, download=download)
        except RuntimeError as exc:
            raise RuntimeError(
                f"\n[D_alpha scoring] Could not load {dataset} from --data_dir={data_dir!r} "
                f"({exc}).\n"
                f"This means the directory doesn't contain the extracted dataset yet — computing "
                f"D_alpha needs the real images to rebuild the gradient trajectory, so this can't "
                f"be skipped. Two fixes:\n"
                f"  1) Add --download to fetch it once (CIFAR-100 is ~170MB; ImageNet needs to "
                f"already be staged, --download won't fetch it).\n"
                f"  2) Point --data_dir at wherever it's already extracted.\n"
                f"Note: train_kl_selection.py / checkpoint_ablation.sh stage the dataset into "
                f"$SLURM_TMPDIR, which is node-local and wiped when that job ends — if you're "
                f"running this aggregator separately, use a persistent path here instead, not "
                f"$SLURM_TMPDIR.\n"
            ) from exc

    def _model_with(self, ckpt_path):
        if self._model is None:
            self._model = self.sc.build_model(self.cfg, self.cfg['num_classes'], self.device)
        self._model.load_state_dict(self.torch.load(ckpt_path, map_location=self.device))
        return self._model

    def _loaders(self, seed):
        if seed not in self._loader_cache:
            self.sc.set_seed(seed)
            _, eval_loader, _, n_train = self.sc.make_loaders(
                self.cfg, seed, self.batch_size, self.num_workers)
            self._loader_cache[seed] = (eval_loader, n_train)
        return self._loader_cache[seed]

    def trajectory(self, seed, k):
        """(LP_list, c_sum, n_train) for the first k checkpoints, on seed's split."""
        key = (seed, k)
        if key in self._traj_cache:
            return self._traj_cache[key]

        eval_loader, n_train = self._loaders(seed)
        ckpt_paths = self.ckpt_paths_all[:k] if k and k > 0 else self.ckpt_paths_all

        R_L = R_Phi = None
        if self.proj_dim > 0:
            R_L   = self.kl.build_sparse_jl(self.cfg['num_classes'], self.proj_dim,
                                            seed=seed).to(self.device)
            R_Phi = self.kl.build_sparse_jl(self.sc.FEAT_DIM, self.proj_dim,
                                            seed=seed + 1).to(self.device)

        LP_list = []
        for ckpt in ckpt_paths:
            model = self._model_with(ckpt)
            L, Phi = self.kl.extract_factored_features(model, eval_loader, self.device,
                                                        R_L=R_L, R_Phi=R_Phi)
            LP_list.append((L, Phi))

        c_sum = self.kl._build_c_sum(LP_list, n_train, self.device)
        self._traj_cache[key] = (LP_list, c_sum, n_train)
        return self._traj_cache[key]

    def d_alpha(self, seed, k, indices_path):
        """D_alpha for the subset saved at indices_path, scored on (seed, k)'s trajectory."""
        LP_list, c_sum, n_train = self.trajectory(seed, k)

        indices = self.torch.load(indices_path, map_location='cpu')
        indices = self.torch.as_tensor(indices, dtype=self.torch.long)
        m = int(indices.numel())
        if m == 0:
            return float('nan')

        alpha = self.torch.zeros(n_train, 1, device=self.device)
        alpha[indices.to(self.device)] = 1.0

        # Identical formula to train_kl_selection.py's own final objective
        # (gradient_ranked_single_swap's h / term1 / term2), just evaluated
        # at an externally-supplied alpha instead of the swap-descent
        # iterate — algebraically equal to (1/m^2) alpha^T P0 alpha + alpha^T q.
        p0_alpha = self.kl._factored_Kv(LP_list, alpha, self.device)
        h = (2.0 / m ** 2) * p0_alpha - (2.0 / (m * n_train)) * c_sum
        mask  = alpha.squeeze() == 1.0
        term1 = 0.5 * h[mask].sum()
        term2 = (1.0 / (m * n_train)) * c_sum[mask].sum()
        return float((term1 - term2).item())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='cifar100', choices=list(N_TRAIN))
    p.add_argument('--fractions', default=[0.1, 0.2, 0.4, 0.5], type=float, nargs='+')
    p.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs='+')
    p.add_argument('--methods', default=None, nargs='+',
                   help='Default: kl_faithful + every method found on disk.')
    p.add_argument('--checkpoints', default=None, type=int, nargs='+',
                   help='Restrict the report to these --num_checkpoints (K) values. '
                        'Default: report every K discovered on disk, plus every '
                        'method that does not tag K at all.')
    p.add_argument('--scratch', default=os.environ.get('SCRATCH', '/home/mila/a/ahmedm/scratch'))
    p.add_argument('--reference_timing', default=None,
                   help='train_time.json of the shared trajectory. Default: '
                        '$SCRATCH/data_selection_swap/<dataset>/checkpoints/train_time.json')

    # D_alpha scoring (all off/no-op unless --data_dir is given).
    p.add_argument('--data_dir', default=None,
                   help='Dataset root. Required to compute the D_alpha column — it rebuilds '
                        'the trajectory kernel from --checkpoint_dir, so it needs the actual '
                        'data. If omitted, D_alpha is left blank and everything else behaves '
                        'exactly as before.')
    p.add_argument('--checkpoint_dir', default=None,
                   help='Reference trajectory checkpoint directory for D_alpha scoring. '
                        'Default: $SCRATCH/data_selection_swap/<dataset>/checkpoints '
                        '(same convention as train_kl_selection.py / checkpoint_ablation.sh).')
    p.add_argument('--skip_objective', action='store_true', default=False,
                   help='Skip D_alpha even if --data_dir is given.')
    p.add_argument('--baseline_score_checkpoints', default=10, type=int,
                   help='K used to score rows whose saved files carry no _K<k> tag, i.e. '
                        'every non-trajectory baseline (herding, moderate coreset, graphcut, '
                        'facility location, EL2N, random).')
    p.add_argument('--objective_batch_size', default=128, type=int)
    p.add_argument('--objective_num_workers', default=4, type=int)
    p.add_argument('--proj_dim', default=0, type=int,
                   help='Optional sparse-JL projection dim for faster, approximate D_alpha '
                        'scoring, matching train_kl_selection.py --proj_dim. 0 = exact.')
    p.add_argument('--download', action='store_true', default=False)
    args = p.parse_args()

    ref_path = args.reference_timing or os.path.join(
        args.scratch, 'data_selection_swap', args.dataset, 'checkpoints', 'train_time.json')
    ref = load_json(ref_path)
    if ref is None:
        print(f"[WARNING] No reference timing at {ref_path}; 'reference' cost will be NaN.")
    epoch_seconds = ref['epoch_seconds'] if ref else None

    checkpoint_dir = args.checkpoint_dir or os.path.join(
        args.scratch, 'data_selection_swap', args.dataset, 'checkpoints')

    scorer = None
    if args.data_dir is not None and not args.skip_objective:
        scorer = ObjectiveScorer(args.dataset, args.data_dir, checkpoint_dir, args.download,
                                 args.objective_batch_size, args.objective_num_workers,
                                 args.proj_dim)
    else:
        print("[INFO] --data_dir not provided (or --skip_objective set): D_alpha column "
              "will be blank. Pass --data_dir to compute it.")

    methods = args.methods or discover_methods(args.scratch, args.dataset)
    warned_methods = set()
    rows = []
    for fraction in args.fractions:
        budget = int(fraction * N_TRAIN[args.dataset])
        for method in methods:
            # K -> accumulated (acc, selection_s, reference_s, total_s, obj) lists across seeds
            by_k = {}
            for seed in args.seeds:
                d = run_dir(args.scratch, method, args.dataset, budget, seed)
                for k, metrics_path, timing_path in discover_runs(d, method, budget):
                    if args.checkpoints is not None and k is not None and k not in args.checkpoints:
                        continue
                    bucket = by_k.setdefault(k, dict(acc=[], sel=[], refc=[], tot=[], obj=[]))

                    metrics = load_json(metrics_path)
                    if metrics:
                        last_epoch = str(max(int(ep) for ep in metrics))
                        bucket['acc'].append(float(metrics[last_epoch]))

                    timing = load_json(timing_path) if timing_path else None
                    if timing:
                        s = timing['selection_seconds']
                        e = timing.get('reference_epochs_used', 0)
                        r = sum(epoch_seconds[:e]) if epoch_seconds else float('nan')
                        bucket['sel'].append(s)
                        bucket['refc'].append(r)
                        bucket['tot'].append(s + r)

                    if scorer is not None:
                        indices_path = indices_path_for(metrics_path)
                        if indices_path is not None:
                            score_k = k if k is not None else args.baseline_score_checkpoints
                            try:
                                bucket['obj'].append(scorer.d_alpha(seed, score_k, indices_path))
                            except Exception as exc:  # keep the report alive on a bad run
                                if method not in warned_methods:
                                    print(f"[WARNING] D_alpha failed for method={method} "
                                          f"seed={seed} K={score_k}: {exc}")
                                    warned_methods.add(method)

            for k, bucket in sorted(by_k.items(), key=lambda kv: (kv[0] is None, kv[0])):
                if not bucket['acc'] and not bucket['sel']:
                    continue
                rows.append(dict(fraction=fraction, method=method, k=k,
                                 n_seeds=len(bucket['acc']), acc=mean_std(bucket['acc']),
                                 selection_s=mean_std(bucket['sel']),
                                 reference_s=mean_std(bucket['refc']),
                                 total_s=mean_std(bucket['tot']),
                                 obj=mean_std(bucket['obj'])))

    header = (f"{'frac':<6}{'method':<34}{'K':<6}{'seeds':<7}{'test acc %':<16}"
              f"{'D_alpha':<20}{'selection (s)':<20}{'reference (s)':<16}{'total (s)':<16}")
    print("\n" + header + "\n" + "-" * len(header))
    last = None
    for r in rows:
        if last is not None and r['fraction'] != last:
            print("-" * len(header))
        last = r['fraction']
        k_disp = str(r['k']) if r['k'] is not None else '–'
        obj_m, obj_s = r['obj']
        obj_disp = f"{obj_m:8.4g} ± {obj_s:<8.4g}" if not np.isnan(obj_m) else f"{'–':<19}"
        print(f"{r['fraction']:<6}{r['method']:<34}{k_disp:<6}{r['n_seeds']:<7}"
              f"{r['acc'][0]:6.2f} ± {r['acc'][1]:<6.2f}"
              f"{obj_disp:<20}"
              f"{r['selection_s'][0]:9.1f} ± {r['selection_s'][1]:<8.1f}"
              f"{r['reference_s'][0]:<16.1f}{r['total_s'][0]:<16.1f}")

    out_csv = os.path.join(args.scratch, 'data_selection_baselines',
                           f'comparison_{args.dataset}.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['fraction', 'method', 'num_checkpoints', 'n_seeds', 'acc_mean', 'acc_std',
                    'D_alpha_mean', 'D_alpha_std',
                    'selection_s_mean', 'selection_s_std', 'reference_s', 'total_s_mean'])
        for r in rows:
            w.writerow([r['fraction'], r['method'], r['k'] if r['k'] is not None else '',
                        r['n_seeds'], *r['acc'], *r['obj'],
                        *r['selection_s'], r['reference_s'][0], r['total_s'][0]])
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()