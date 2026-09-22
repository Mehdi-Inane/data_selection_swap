import argparse
import json
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dotmap import DotMap

from cords.utils.data.dataloader.SL.adaptive import GradMatchDataLoader
from cords.utils.models import ResNet18


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # ── 0. Argument Parsing ───────────────────────────────────────────────────────
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fraction', default=0.3, type=float)
    p.add_argument('-seed', '--seed', default=42, type=int)
    args = p.parse_args()

    set_seed(args.seed)

    # ── 1. Data Preparation & Split ──────────────────────────────────────────────
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
    budget  = int(args.fraction * n_train)
    
    # ── CHANGED: seed sub-directory so runs never overwrite each other ────────────
    base_save_directory = f"/home/mila/a/ahmedm/scratch/gradmatch_swap/cifar100/{budget}/seed_{args.seed}"
    os.makedirs(base_save_directory, exist_ok=True)

    # ── 2. Logging & Artifact File Setup ─────────────────────────────────────────
    LOG_FILE = os.path.join(base_save_directory, f"gradmatch_{budget}_training.log")
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

    MODEL_SAVE_PATH   = os.path.join(base_save_directory, f"gradmatch_{budget}_model.pth")
    INDICES_SAVE_PATH = os.path.join(base_save_directory, f"gradmatch_{budget}_indices.pt")
    WEIGHTS_SAVE_PATH = os.path.join(base_save_directory, f"gradmatch_{budget}_weights.pt")
    METRICS_SAVE_PATH = os.path.join(base_save_directory, f"gradmatch_{budget}_metrics.json")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Running GradMatch training on device: {device}")

    # Reproducible train/val split using command-line seed
    trainset, valset = random_split(
        full_train, 
        [n_train, n_val], 
        generator=torch.Generator().manual_seed(args.seed)
    )

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    valloader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    testloader  = DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # ── 3. Model Setup (Using CORDS ResNet18) ────────────────────────────────────
    num_classes = 100 if 'cifar100' in base_save_directory else 10
    model = ResNet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    criterion_nored = nn.CrossEntropyLoss(reduction='none')

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ── 4. GradMatch DataLoader ───────────────────────────────────────────────────
    dss_args = DotMap({
        'type'           : 'GradMatch',
        'model'          : model,
        'loss'           : criterion_nored,
        'eta'            : LR,
        'num_classes'    : num_classes,
        'num_epochs'     : NUM_EPOCHS,
        'device'         : device,
        'valid'          : False,
        'fraction'       : args.fraction,
        'select_every'   : 20,
        'kappa'          : 0,
        'linear_layer'   : False,
        'selection_type' : 'PerClassPerGradient',
        'greedy'         : 'Stochastic',
        'collate_fn'     : None,
        'v1'             : True,
        'lam'            : 0.5,
        'eps'            : 1e-100,
    })

    dataloader = GradMatchDataLoader(
        trainloader, valloader, dss_args, logger,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        pin_memory  = True,
    )

    # ── 5. Training Loop ──────────────────────────────────────────────────────────
    logger.info("Starting training loop...")
    accuracy_history = {}

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for inputs, targets, weights in dataloader:
            inputs, targets, weights = (inputs.to(device),
                                        targets.to(device),
                                        weights.to(device))
            optimizer.zero_grad()
            outputs = model(inputs)
            losses  = criterion_nored(outputs, targets)
            loss    = torch.dot(losses, weights / weights.sum())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        if epoch % 50 == 0 or epoch == NUM_EPOCHS:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in testloader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x).argmax(1)
                    correct += pred.eq(y).sum().item()
                    total   += y.size(0)

            acc = 100.0 * correct / total
            avg_loss = running_loss / len(dataloader)
            accuracy_history[epoch] = acc

            logger.info(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Test Acc: {acc:.2f}% | Train Loss: {avg_loss:.4f}")

    # ── 6. Save Artifacts ─────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Saved trained model weights to {MODEL_SAVE_PATH}")

    idxes = torch.tensor(dataloader.subset_indices)
    gammas = torch.tensor(dataloader.subset_weights)
    torch.save(idxes, INDICES_SAVE_PATH)
    torch.save(gammas, WEIGHTS_SAVE_PATH)
    logger.info(f"Saved selected indices to {INDICES_SAVE_PATH} and weights to {WEIGHTS_SAVE_PATH}")

    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(accuracy_history, f, indent=4)
    logger.info(f"Saved accuracy log history to {METRICS_SAVE_PATH}")


if __name__ == "__main__":
    main()