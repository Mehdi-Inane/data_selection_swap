import argparse
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
    p.add_argument('--dataset', default='cifar100', choices=['cifar100', 'imagenet'])
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
    cfg = common.get_dataset_config(args.dataset, args.data_dir, download=args.download)
    full_train = cfg['full_train']

    n_val   = int(0.1 * len(full_train))
    n_train = len(full_train) - n_val

    # Ensure the exact same trainset is used to compute the trajectory
    trainset, valset = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.num_workers
    )

    # ── 3. Model & Optimizer Setup ─────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = cfg['num_classes']
    NUM_EPOCHS = cfg['num_epochs']

    model = common.build_model(cfg, NUM_CLASSES, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    logger.info(f"Starting full-dataset training on {device} for {NUM_EPOCHS} epochs...")

    # ── 4. Training Loop & Checkpoint Saving ───────────────────────────────────
    # Per-epoch wall-clock lets analysis.comparison charge each
    # selection method for the prefix of this trajectory it consumes.
    epoch_seconds = []
    epoch_lrs = []
    checkpoint_records = []
    global_step = 0
    steps_per_epoch = len(trainloader)
    timing_path = os.path.join(args.save_dir, "train_time.json")
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_lr = float(optimizer.param_groups[0]['lr'])
        epoch_lrs.append(epoch_lr)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            global_step += 1

            running_loss += loss.item()

        scheduler.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        epoch_seconds.append(time.perf_counter() - t0)
        with open(timing_path, 'w') as fh:
            json.dump({"epoch_seconds": epoch_seconds,
                       "total_seconds": sum(epoch_seconds),
                       "device": torch.cuda.get_device_name() if device == 'cuda' else 'cpu'},
                      fh, indent=4)
        avg_loss = running_loss / len(trainloader)

        # Logging
        if epoch % 10 == 0 or epoch <= 10:
            logger.info(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f}")

        # Checkpoint Saving Logic: epochs 1 to 10, then every 10 epochs
        if epoch <= 10 or epoch % 10 == 0:
            # Zero-pad epoch number so glob.glob() sorts them correctly alphabetically
            save_path = os.path.join(args.save_dir, f"checkpoint_{epoch:03d}.pth")
            torch.save(model.state_dict(), save_path)
            checkpoint_records.append({
                "file": os.path.basename(save_path),
                "step": global_step,
                "epoch": epoch,
                "lr": epoch_lr,
            })
            logger.info(f"--> Saved checkpoint: {save_path}")

        # Keep bare state_dict checkpoints for all baseline loaders, and store
        # optimizer metadata in a sidecar manifest. This records eta_k without
        # changing the checkpoint format expected by older experiment code.
        manifest = {
            "optimizer": "sgd",
            "lr": args.lr,
            "weight_decay": 5e-4,
            "dataset": args.dataset,
            "seed": args.seed,
            "num_epochs": NUM_EPOCHS,
            "batch_size": args.batch_size,
            "n_train": n_train,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": NUM_EPOCHS * steps_per_epoch,
            "epochs_completed": epoch,
            "epoch_lrs": epoch_lrs,
            "checkpoints": checkpoint_records,
        }
        with open(os.path.join(args.save_dir, trajectory.MANIFEST_NAME), 'w') as fh:
            json.dump(manifest, fh, indent=4)

    logger.info("Finished trajectory training.")

if __name__ == "__main__":
    main()
