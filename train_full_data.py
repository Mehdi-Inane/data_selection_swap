import argparse
import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Assuming you are using the same ResNet18 from cords
from cords.utils.models import ResNet18

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    p = argparse.ArgumentParser(description="Train full CIFAR-10 reference model for KL-Faithful selection")
    p.add_argument('-seed', '--seed', default=42, type=int)
    p.add_argument('--save_dir', type=str, 
                   default='/home/mila/a/ahmedm/scratch/data_selection_swap/cifar100/checkpoints',
                   help='Directory to save model checkpoints')
    args = p.parse_args()

    set_seed(args.seed)

    # ── 1. Directory & Logging Setup ──────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Saving checkpoints to: {args.save_dir}")

    # ── 2. Data Preparation & Split ────────────────────────────────────────────
    # Must perfectly match the split used in data selection
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    full_train = datasets.CIFAR100(root='data/', train=True, download=True, transform=transform_train)

    n_val   = int(0.1 * len(full_train))
    n_train = len(full_train) - n_val

    # Ensure the exact same 45k trainset is used to compute the trajectory
    trainset, valset = random_split(
        full_train, 
        [n_train, n_val], 
        generator=torch.Generator().manual_seed(args.seed)
    )

    BATCH_SIZE = 128
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=1)

    # ── 3. Model & Optimizer Setup ─────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    num_classes = 100 if 'cifar100' in args.save_dir else 10
    print(num_classes)

    model = ResNet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    NUM_EPOCHS = 350
    LR = 0.01
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    logger.info(f"Starting full-dataset training on {device} for {NUM_EPOCHS} epochs...")

    # ── 4. Training Loop & Checkpoint Saving ───────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(trainloader)

        # Logging
        if epoch % 10 == 0 or epoch <= 10:
            logger.info(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f}")

        # Checkpoint Saving Logic: epochs 1 to 10, then every 10 epochs
        if epoch <= 10 or epoch % 10 == 0:
            # Zero-pad epoch number so glob.glob() sorts them correctly alphabetically
            save_path = os.path.join(args.save_dir, f"checkpoint_{epoch:03d}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"--> Saved checkpoint: {save_path}")

    logger.info("Finished trajectory training.")

if __name__ == "__main__":
    main()