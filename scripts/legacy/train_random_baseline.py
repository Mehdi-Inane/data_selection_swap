import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fraction', default=0.3, type=float)
    p.add_argument('-seed', '--seed', default=42, type=int)
    args = p.parse_args()
    
    set_seed(args.seed)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    full_train = datasets.CIFAR100(root='data/', train=True, download=True, transform=transform_train)
    testset    = datasets.CIFAR100(root='data/', train=False, download=True, transform=transform_test)

    n_val   = int(0.1 * len(full_train))
    n_train = len(full_train) - n_val

    budget = int(args.fraction * n_train)

    # ── CHANGED: seed sub-directory so runs never overwrite each other ────────────
    base_save_directory = f"/home/mila/a/ahmedm/scratch/gradmatch_swap/cifar100/{budget}/seed_{args.seed}"
    os.makedirs(base_save_directory, exist_ok=True)
    
    LOG_FILE = os.path.join(base_save_directory, f"random_{budget}_training.log")
    
    # ── 1. Logging & Configuration Setup ──────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    BATCH_SIZE = 128
    NUM_EPOCHS = 350
    LR = 0.01
    NUM_SAMPLES = budget
    SEED = args.seed

    MODEL_SAVE_PATH   = os.path.join(base_save_directory, f"random_{budget}_model.pth")
    INDICES_SAVE_PATH = os.path.join(base_save_directory, f"random_{budget}_indices.pt")
    METRICS_SAVE_PATH = os.path.join(base_save_directory, f"random_{budget}_metrics.json")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Running baseline on device: {device}")

    # ── 2. Data Preparation & Sampling ────────────────────────────────────────────
    # Match train/val split (45,000 / 5,000)
    trainset, _ = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    # Sample budget indices uniformly at random
    g_sample = torch.Generator().manual_seed(SEED)
    random_indices = torch.randperm(len(trainset), generator=g_sample)[:NUM_SAMPLES]

    # Save sampled indices
    torch.save(random_indices, INDICES_SAVE_PATH)
    logger.info(f"Sampled {NUM_SAMPLES} random indices and saved to {INDICES_SAVE_PATH}")

    random_subset = Subset(trainset, random_indices)
    trainloader = DataLoader(random_subset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=1)
    testloader  = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=1)

    # ── 3. Model Setup ────────────────────────────────────────────────────────────
    num_classes = 100 if 'cifar100' in base_save_directory else 10
    model = torchvision.models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ── 4. Training Loop ──────────────────────────────────────────────────────────
    logger.info("Starting training loop...")
    accuracy_history = {}

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

        # Log metrics every 50 epochs and at end
        if epoch % 50 == 0 or epoch == NUM_EPOCHS:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in testloader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x).argmax(1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)
                    
            acc = 100.0 * correct / total
            avg_loss = running_loss / len(trainloader)
            accuracy_history[epoch] = acc
            
            logger.info(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Test Acc: {acc:.2f}% | Train Loss: {avg_loss:.4f}")

    # ── 5. Save Artifacts ─────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Saved trained model weights to {MODEL_SAVE_PATH}")

    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(accuracy_history, f, indent=4)
    logger.info(f"Saved accuracy log history to {METRICS_SAVE_PATH}")


if __name__ == "__main__":
    main()