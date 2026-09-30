import argparse
import copy
import json
import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from .. import common
from .. import trajectory

def set_seed(seed=42):
    """Backward-compatible alias for the shared seed helper."""
    common.set_seed(seed)

def get_dataset_config(dataset: str, data_dir: str, download: bool = False) -> dict:
    """Backward-compatible alias for the shared dataset configuration."""
    return common.get_dataset_config(dataset, data_dir, download=download)

def main():
    p = argparse.ArgumentParser(description="Train full reference model for KL-Faithful selection")
    p.add_argument('--dataset', default='cifar100', choices=list(common.DATASETS))
    p.add_argument('--architecture', default=None, choices=list(common.ARCHITECTURES),
                   help='Default: resnet18, or lenet for MNIST.')
    p.add_argument('--data_dir', required=True, type=str,
                   help='Root of the staged dataset (e.g. $SLURM_TMPDIR/cifar100_data)')
    p.add_argument('-seed', '--seed', default=42, type=int)
    p.add_argument('--save_dir', type=str, required=True,
                   help='Directory to save model checkpoints')
    p.add_argument('--download', action='store_true', default=False,
                   help='Download dataset if not present')
    p.add_argument('--batch_size', default=128, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--lr', default=0.01, type=float)
    p.add_argument('--num_epochs', default=None, type=int)
    p.add_argument('--resume', action='store_true',
                   help='Resume from training_state.pth after validating the run config.')
    args = p.parse_args()

    common.set_seed(args.seed)

    # ── 1. Directory & Logging Setup ──────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.save_dir, "train.log"))]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Dataset={args.dataset} | Saving checkpoints to: {args.save_dir}")

    # ── 2. Data Preparation & Split ────────────────────────────────────────────
    cfg = common.get_dataset_config(args.dataset, args.data_dir, download=args.download,
                                    architecture=args.architecture)
    full_train = cfg['full_train']

    n_val   = int(0.1 * len(full_train))
    n_train = len(full_train) - n_val

    # Ensure the exact same trainset is used to compute the trajectory
    trainset, valset = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    pool_indices_sha256 = common.ordered_indices_fingerprint(trainset.indices)

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.num_workers
    )

    # ── 3. Model & Optimizer Setup ─────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = cfg['num_classes']
    NUM_EPOCHS = args.num_epochs or cfg['num_epochs']

    model = common.build_model(cfg, NUM_CLASSES, device)
    initial_state_sha256 = common.state_dict_fingerprint(model.state_dict())

    criterion = nn.CrossEntropyLoss()
    optimizer = common.build_optimizer(model.parameters(), 'sgd', args.lr,
                                       weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    immutable_config = {
        'schema_version': 2,
        'dataset': args.dataset,
        'architecture': cfg['architecture'],
        'seed': args.seed,
        'optimizer': 'sgd',
        'lr': args.lr,
        'weight_decay': 5e-4,
        'num_epochs': NUM_EPOCHS,
        'batch_size': args.batch_size,
        'n_train': n_train,
        'steps_per_epoch': len(trainloader),
        'pool_indices_sha256': pool_indices_sha256,
        'initial_state_sha256': initial_state_sha256,
    }
    run_id = common.config_fingerprint(immutable_config)
    state_path = os.path.join(args.save_dir, 'training_state.pth')
    manifest_path = os.path.join(args.save_dir, trajectory.MANIFEST_NAME)

    logger.info(f"Starting full-dataset training on {device} for {NUM_EPOCHS} epochs...")

    # ── 4. Training Loop & Checkpoint Saving ───────────────────────────────────
    # Per-epoch wall-clock lets analysis.comparison charge each
    # selection method for the prefix of this trajectory it consumes.
    epoch_seconds = []
    epoch_lrs = []
    checkpoint_records = []
    global_step = 0
    start_epoch = 1
    steps_per_epoch = len(trainloader)
    timing_path = os.path.join(args.save_dir, "train_time.json")

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
        epoch_seconds = list(state['epoch_seconds'])
        epoch_lrs = list(state['epoch_lrs'])
        checkpoint_records = list(state['checkpoints'])
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

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_lr = float(optimizer.param_groups[0]['lr'])
        epoch_lrs.append(epoch_lr)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0

        save_this_epoch = epoch <= 10 or epoch % 10 == 0 or epoch == NUM_EPOCHS
        checkpoint_snapshot = None
        checkpoint_step = None
        for batch_index, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # Pair theta_s with the LR and gradient used for the outgoing
            # update, instead of tagging a post-epoch state with the incoming
            # epoch LR.  This is the same convention as training.reference.
            if save_this_epoch and batch_index == steps_per_epoch - 1:
                checkpoint_snapshot = copy.deepcopy(model.state_dict())
                checkpoint_step = global_step + 1
            optimizer.step()
            global_step += 1

            running_loss += loss.item()

        scheduler.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        epoch_seconds.append(time.perf_counter() - t0)
        common.atomic_json_dump(
            {"epoch_seconds": epoch_seconds,
             "total_seconds": sum(epoch_seconds),
             "device": torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
             "run_id": run_id},
            timing_path)
        avg_loss = running_loss / len(trainloader)

        # Logging
        if epoch % 10 == 0 or epoch <= 10:
            logger.info(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f}")

        # Checkpoint Saving Logic: epochs 1 to 10, then every 10 epochs
        if save_this_epoch:
            if checkpoint_snapshot is None:
                raise RuntimeError("Failed to capture the checkpoint update boundary")
            # Zero-pad epoch number so glob.glob() sorts them correctly alphabetically
            save_path = os.path.join(args.save_dir, f"checkpoint_{epoch:03d}.pth")
            common.atomic_torch_save(checkpoint_snapshot, save_path)
            checkpoint_records.append({
                "file": os.path.basename(save_path),
                "step": checkpoint_step,
                "epoch": epoch,
                "lr": epoch_lr,
                "state_convention": "pre_outgoing_update",
            })
            logger.info(f"--> Saved checkpoint: {save_path}")

        # Keep bare state_dict checkpoints for all baseline loaders, and store
        # optimizer metadata in a sidecar manifest. This records eta_k without
        # changing the checkpoint format expected by older experiment code.
        manifest = {
            "schema_version": 2,
            "run_id": run_id,
            "optimizer": "sgd",
            "lr": args.lr,
            "weight_decay": 5e-4,
            "dataset": args.dataset,
            "architecture": cfg['architecture'],
            "seed": args.seed,
            "num_epochs": NUM_EPOCHS,
            "batch_size": args.batch_size,
            "n_train": n_train,
            "steps_per_epoch": steps_per_epoch,
            "pool_indices_sha256": pool_indices_sha256,
            "total_steps": NUM_EPOCHS * steps_per_epoch,
            "epochs_completed": epoch,
            "epoch_lrs": epoch_lrs,
            "checkpoints": checkpoint_records,
            "immutable_config": immutable_config,
            "initial_state_sha256": initial_state_sha256,
        }
        common.atomic_json_dump(manifest, manifest_path)

        training_state = {
            'run_id': run_id,
            'initial_state_sha256': initial_state_sha256,
            'epoch': epoch,
            'global_step': global_step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch_seconds': epoch_seconds,
            'epoch_lrs': epoch_lrs,
            'checkpoints': checkpoint_records,
            'pool_indices_sha256': pool_indices_sha256,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state_all': torch.cuda.get_rng_state_all() if device == 'cuda' else None,
        }
        state_tmp = f"{state_path}.tmp.{os.getpid()}"
        torch.save(training_state, state_tmp)
        os.replace(state_tmp, state_path)

    logger.info("Finished trajectory training.")

if __name__ == "__main__":
    main()
