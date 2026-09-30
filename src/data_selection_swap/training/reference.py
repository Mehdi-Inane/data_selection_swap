#!/usr/bin/env python3
"""Reference trajectory for the selection objective, with the optimizer state
the Adam-weighted objective needs.

Why this trainer exists
──────────────────────────────────
``training.full_data`` keeps bare model state dictionaries for compatibility
and records SGD learning rates in a sidecar manifest. That is enough for
--optimizer uniform and --optimizer sgd, but not for --optimizer adam, which
also needs the bias-corrected second moment vhat_s. This trainer saves that
state alongside each model snapshot.

Checkpoint alignment
────────────────────
At step s the run computes g_s at theta_s, folds it into v_s, then takes
theta_{s+1} = theta_s - eta_s * D_s * mhat_s. The update *from* theta_s
therefore uses v_s, so a conventional checkpoint written after
optimizer.step() pairs theta_{s+1} with v_s and is off by one. Here theta
is snapshotted immediately *before* the step and vhat immediately after,
and both go in the same record with the step number and eta_s.

Record layout
─────────────
    {'model': state_dict at theta_s,
     'exp_avg_sq': {param name: v_s} or None,
     'step': s, 'epoch': e, 'lr': eta_s,
     'optimizer': 'adam', 'betas': (b1, b2), 'eps': eps}

plus a trajectory.json manifest holding steps_per_epoch, the per-epoch
learning rates (CosineAnnealingLR steps once per epoch, so eta is constant
within an epoch and this determines every eta_s), and the checkpoint list.
Note these records are not bare state_dicts: read them through
``trajectory.load_weights()``, as all maintained selectors in this repository
do.

Usage:
    python scripts/train/reference.py --dataset cifar100 --data_dir ... \\
        --save_dir $SCRATCH/data_selection_swap/cifar100/adam_ref \\
        --optimizer adam --lr 1e-3 --num_checkpoints 10
"""

import argparse
import copy
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import common
from .. import trajectory as ref


def choose_epochs(num_epochs, num_checkpoints, spacing, explicit):
    """Checkpoint epochs spanning the whole run, not a prefix of it."""
    if explicit:
        bad = [e for e in explicit if not 1 <= e <= num_epochs]
        if bad:
            raise ValueError(f"--checkpoint_epochs {bad} outside 1..{num_epochs}")
        return sorted(set(explicit))
    if num_checkpoints >= num_epochs:
        return list(range(1, num_epochs + 1))
    if spacing == 'linear':
        grid = np.linspace(1, num_epochs, num_checkpoints)
    elif spacing == 'log':
        grid = np.geomspace(1, num_epochs, num_checkpoints)
    else:
        raise ValueError(f"Unknown spacing: {spacing!r}")
    return sorted(set(int(round(e)) for e in grid))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='cifar100', choices=list(common.DATASETS))
    p.add_argument('--architecture', default=None, choices=list(common.ARCHITECTURES),
                   help='Default: resnet18, or lenet for MNIST.')
    p.add_argument('--data_dir', required=True, type=str)
    p.add_argument('--save_dir', required=True, type=str)
    p.add_argument('--seed', '-seed', default=42, type=int)
    p.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'adamw'])
    p.add_argument('--lr', default=None, type=float,
                   help='Default: 0.01 for sgd, 1e-3 for adam/adamw.')
    p.add_argument('--weight_decay', default=5e-4, type=float)
    p.add_argument('--beta1', default=0.9, type=float)
    p.add_argument('--beta2', default=0.999, type=float)
    p.add_argument('--adam_eps', default=1e-8, type=float)
    p.add_argument('--batch_size', default=128, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--num_epochs', default=None, type=int,
                   help="Default: the dataset's own (300 for CIFAR-100).")
    p.add_argument('--num_checkpoints', default=10, type=int)
    p.add_argument('--spacing', default='linear', choices=['linear', 'log'],
                   help='How checkpoint epochs are spread over the run.')
    p.add_argument('--checkpoint_epochs', default=None, type=int, nargs='+',
                   help='Explicit checkpoint epochs; overrides --num_checkpoints.')
    p.add_argument('--download', action='store_true', default=False)
    p.add_argument('--resume', action='store_true',
                   help='Resume from training_state.pth after validating the run config.')
    args = p.parse_args()

    common.set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(args.save_dir, 'train.log'))])
    logger = logging.getLogger(__name__)

    cfg = common.get_dataset_config(args.dataset, args.data_dir, download=args.download,
                                    architecture=args.architecture)
    num_epochs = args.num_epochs or cfg['num_epochs']
    lr = args.lr if args.lr is not None else (0.01 if args.optimizer == 'sgd' else 1e-3)

    # Same split as every selection script, so the trajectory and the
    # selection pool are the same 90%.
    trainset, _, _, n_train = common.make_loaders(
        cfg, args.seed, args.batch_size, args.num_workers)
    pool_indices_sha256 = common.ordered_indices_fingerprint(trainset.indices)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=args.num_workers)

    spe = ref.steps_per_epoch(n_train, args.batch_size)
    if len(trainloader) != spe:
        logger.warning(f"steps_per_epoch mismatch: dataloader gives {len(trainloader)}, "
                       f"ceil(n/bs) gives {spe}; using the dataloader's value.")
        spe = len(trainloader)

    ckpt_epochs = choose_epochs(num_epochs, args.num_checkpoints, args.spacing,
                                args.checkpoint_epochs)
    ckpt_steps = {e * spe: e for e in ckpt_epochs}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = common.build_model(cfg, cfg['num_classes'], device)
    initial_state_sha256 = common.state_dict_fingerprint(model.state_dict())
    criterion = nn.CrossEntropyLoss()
    optimizer = ref.build_optimizer(model.parameters(), args.optimizer, lr,
                                    weight_decay=args.weight_decay,
                                    betas=(args.beta1, args.beta2), eps=args.adam_eps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    immutable_config = {
        'schema_version': 2,
        'dataset': args.dataset,
        'architecture': cfg['architecture'],
        'seed': args.seed,
        'optimizer': args.optimizer,
        'lr': lr,
        'weight_decay': args.weight_decay,
        'betas': [args.beta1, args.beta2],
        'eps': args.adam_eps,
        'num_epochs': num_epochs,
        'batch_size': args.batch_size,
        'n_train': n_train,
        'steps_per_epoch': spe,
        'pool_indices_sha256': pool_indices_sha256,
        'checkpoint_epochs': ckpt_epochs,
        'initial_state_sha256': initial_state_sha256,
    }
    run_id = common.config_fingerprint(immutable_config)
    state_path = os.path.join(args.save_dir, 'training_state.pth')
    manifest_path = os.path.join(args.save_dir, ref.MANIFEST_NAME)

    logger.info(f"dataset={args.dataset} optimizer={args.optimizer} lr={lr} "
                f"epochs={num_epochs} n_train={n_train} steps/epoch={spe} "
                f"total steps={num_epochs * spe}")
    logger.info(f"checkpoint epochs: {ckpt_epochs}")

    names = [n for n, _ in model.named_parameters()]
    epoch_lrs, epoch_seconds, records = [], [], []
    global_step = 0
    start_epoch = 1

    if args.resume:
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"--resume requested but {state_path} does not exist")
        state = torch.load(state_path, map_location=device)
        if state.get('run_id') != run_id:
            raise ValueError(
                f"Resume config mismatch: requested run {run_id}, state belongs to "
                f"{state.get('run_id')}. Use a new --save_dir."
            )
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = int(state['epoch']) + 1
        global_step = int(state['global_step'])
        epoch_lrs = list(state['epoch_lrs'])
        epoch_seconds = list(state['epoch_seconds'])
        records = list(state['checkpoints'])
        torch.set_rng_state(state['torch_rng_state'].cpu())
        if device == 'cuda' and state.get('cuda_rng_state_all') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng_state_all'])
        logger.info(f"Resuming run {run_id} at epoch {start_epoch}")
    elif os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            previous = json.load(fh)
        if previous.get('checkpoints'):
            raise FileExistsError(
                f"{args.save_dir} already contains trajectory artifacts. Refusing to "
                "mix runs; pass --resume with the identical config or use a new directory."
            )

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_lr = optimizer.param_groups[0]['lr']
        epoch_lrs.append(float(epoch_lr))
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        model.train()
        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()

            global_step += 1
            snapshot = None
            if global_step in ckpt_steps:
                # theta_s, before the update it is about to take.
                snapshot = copy.deepcopy(model.state_dict())

            optimizer.step()
            running_loss += loss.item()

            if snapshot is not None:
                # v_s, after the update, i.e. the moment that update used.
                exp_avg_sq = None
                if args.optimizer in ('adam', 'adamw'):
                    state = optimizer.state
                    exp_avg_sq = {n: state[p]['exp_avg_sq'].detach().cpu().clone()
                                  for n, p in model.named_parameters() if p in state}
                    missing = [n for n in names if n not in exp_avg_sq]
                    if missing:
                        raise RuntimeError(f"No Adam state for {missing[:3]}... at step "
                                           f"{global_step}; cannot build M_k.")
                e = ckpt_steps[global_step]
                fname = f"checkpoint_{e:03d}.pth"
                common.atomic_torch_save(
                    {'model': {k: v.detach().cpu().clone()
                               for k, v in snapshot.items()},
                     'exp_avg_sq': exp_avg_sq,
                     'step': global_step, 'epoch': e, 'lr': float(epoch_lr),
                     'optimizer': args.optimizer,
                     'betas': (args.beta1, args.beta2), 'eps': args.adam_eps},
                    os.path.join(args.save_dir, fname))
                records.append({'file': fname, 'step': global_step, 'epoch': e,
                                'lr': float(epoch_lr)})
                logger.info(f"--> checkpoint epoch {e} (step {global_step}, "
                            f"lr {epoch_lr:.6f}) -> {fname}")

        scheduler.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        epoch_seconds.append(time.perf_counter() - t0)

        manifest = {
            'schema_version': 2, 'run_id': run_id,
            'optimizer': args.optimizer, 'lr': lr, 'weight_decay': args.weight_decay,
            'betas': [args.beta1, args.beta2], 'eps': args.adam_eps,
            'dataset': args.dataset, 'architecture': cfg['architecture'], 'seed': args.seed,
            'num_epochs': num_epochs, 'batch_size': args.batch_size,
            'n_train': n_train, 'steps_per_epoch': spe,
            'pool_indices_sha256': pool_indices_sha256,
            'total_steps': num_epochs * spe,
            'epochs_completed': epoch, 'immutable_config': immutable_config,
            'initial_state_sha256': initial_state_sha256,
            'epoch_lrs': epoch_lrs, 'checkpoints': records,
        }
        common.atomic_json_dump(manifest, os.path.join(args.save_dir, ref.MANIFEST_NAME))
        common.atomic_json_dump(
            {'epoch_seconds': epoch_seconds,
             'total_seconds': sum(epoch_seconds),
             'device': torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
             'run_id': run_id},
            os.path.join(args.save_dir, 'train_time.json'))

        training_state = {
            'run_id': run_id,
            'initial_state_sha256': initial_state_sha256,
            'epoch': epoch,
            'global_step': global_step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch_lrs': epoch_lrs,
            'epoch_seconds': epoch_seconds,
            'checkpoints': records,
            'pool_indices_sha256': pool_indices_sha256,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state_all': torch.cuda.get_rng_state_all() if device == 'cuda' else None,
        }
        state_tmp = f"{state_path}.tmp.{os.getpid()}"
        torch.save(training_state, state_tmp)
        os.replace(state_tmp, state_path)

        if epoch % 10 == 0 or epoch <= 10:
            logger.info(f"Epoch {epoch:03d}/{num_epochs} | lr {epoch_lr:.6f} | "
                        f"Train Loss: {running_loss / len(trainloader):.4f}")

    # epoch_lrs must cover every epoch for the step-weight sums to be complete.
    if len(epoch_lrs) != num_epochs:
        logger.warning(f"Wrote {len(epoch_lrs)} epoch learning rates for {num_epochs} "
                       f"epochs; the manifest describes a truncated run.")
    logger.info(f"Finished. {len(records)} checkpoints + {ref.MANIFEST_NAME} in "
                f"{args.save_dir}")


if __name__ == "__main__":
    main()
