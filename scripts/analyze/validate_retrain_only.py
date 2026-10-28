"""Dry-run validation of --retrain_only artifact discovery for the 9 recovery cells."""
import glob as globmod
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ.setdefault('SCRATCH', '/network/scratch/a/ahmedm')

from data_selection_swap import trajectory as ref
from data_selection_swap.paths import storage_variant, tree

SCRATCH = os.environ['SCRATCH']
DATASETS = ['cifar10', 'cifar100']
ARCHS = ['resnet18', 'vgg16', 'inception']
KS = [1, 4, 10, 15, 20]
FRACTIONS = [0.1, 0.2, 0.4, 0.5]
SEEDS = [42, 43, 44, 45, 46]
N = len(SEEDS)


def decode(tid):
    i = tid
    seed = SEEDS[i % N]; i //= N
    frac = FRACTIONS[i % len(FRACTIONS)]; i //= len(FRACTIONS)
    K = KS[i % len(KS)]; i //= len(KS)
    arch = ARCHS[i % len(ARCHS)]; i //= len(ARCHS)
    return DATASETS[i], arch, frac, K, seed


tag = ref.objective_tag('sgd', 'auto', 'point', 'persistent')
print(f"objective_tag -> {tag!r}")
variant = storage_variant('sgd', 'sgd')
fail = 0
for tid in [74, 267, 268, 350, 353, 547, 552, 553, 561]:
    ds, arch, frac, K, seed = decode(tid)
    budget = int(frac * 50000)
    base = os.path.join(tree(SCRATCH, 'gradmatch_swap', variant),
                        ds, arch, str(budget), f'seed_{seed}')
    method = f"kl_faithful_full_{arch}{tag}"
    pat = os.path.join(base, f"{method}_{budget}_K{K}_cfg*_indices.pt")
    hits = sorted(globmod.glob(pat))
    run_jsons = sorted(globmod.glob(os.path.join(base, f"{method}_{budget}_K{K}_cfg*_run.json")))
    ok = len(hits) == 1
    extra = ''
    if ok:
        prefix = hits[0][:-len('_indices.pt')]
        met = os.path.exists(prefix + '_metrics.json')
        status = None
        if len(run_jsons) == 1:
            with open(run_jsons[0]) as fh:
                rec = json.load(fh)
            status = rec.get('status')
            args_rec = (rec.get('config') or {}).get('arguments') or {}
            extra = (f" lr={args_rec.get('lr')} bs={args_rec.get('batch_size')}"
                     f" retrain_opt={args_rec.get('retrain_optimizer')}"
                     f" init={os.path.basename(str(args_rec.get('init_ckpt')))}")
        ok = (not met) and status == 'configured'
        extra += f" metrics_exists={met} run_status={status}"
    print(f"{tid:3d} {ds:8s} {arch:9s} f={frac} K={K:2d} s{seed}: "
          f"{len(hits)} indices{' OK' if ok else ' PROBLEM'}{extra}")
    fail += (not ok)
print('ALL-OK' if fail == 0 else f'{fail} PROBLEMS')
