import argparse
import json
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import glob
from cords.utils.models import ResNet18


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_gradient_features(model, dataloader, device):
    """
    Extracts exact last-layer loss gradient representations for all samples.
    Returns G in R^{n x d}, where d = feature_dim * num_classes.
    """
    model.eval()
    feature_list = []
    
    features_dict = {}
    def hook(module, input, output):
        features_dict['linear_in'] = input[0]

    handle = model.linear.register_forward_hook(hook)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            targets_onehot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
            grad_logits = (probs - targets_onehot)  
            
            phi = features_dict['linear_in']
            grad_w = torch.bmm(grad_logits.unsqueeze(2), phi.unsqueeze(1)).reshape(inputs.size(0), -1)
            
            feature_list.append(grad_w.cpu())

    handle.remove()
    # Return unnormalized G directly to preserve exact distance metrics
    return torch.cat(feature_list, dim=0)


def load_trajectory_features(model, dataloader, checkpoint_paths, device):
    """
    Iterates through saved theta_i checkpoints, extracting G_i for each.
    Returns a list of G matrices [G_1, G_2, ..., G_T].
    """
    G_list = []
    for ckpt_path in checkpoint_paths:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        G_i = extract_gradient_features(model, dataloader, device)
        G_list.append(G_i)
    return G_list


def compute_h(G_list, alpha_vec, m, n, c_sum, device):
    """
    Computes h(alpha) across all T checkpoints iteratively to save memory.
    """
    # P_0 @ alpha = \sum_k G_k @ (G_k^T @ alpha)
    p0_alpha = torch.zeros(n, 1, device=device)
    for G in G_list:
        G = G.to(device)
        p0_alpha += G @ (G.T @ alpha_vec)
        
    return (2.0 / (m ** 2)) * p0_alpha - (2.0 / (m * n)) * c_sum


def gradient_ranked_single_swap(G_list, m, max_iters=5000, log_freq=5, device='cuda'):
    """
    Algorithm 1: Gradient-Ranked Single-Swap Descent across T checkpoints.
    Includes tracking of the exact objective value D(alpha_t).
    """
    n = G_list[0].shape[0]
    
    # Pre-compute static vector c = P_0 @ 1_n = \sum_k G_k @ (G_k^T @ 1_n)
    ones_n = torch.ones(n, 1, device=device)
    c_sum = torch.zeros(n, 1, device=device)
    for G in G_list:
        G = G.to(device)
        c_sum += G @ (G.T @ ones_n)

    # Step 1: Draw alpha_0 ~ Unif(B_m)
    perm = torch.randperm(n, device=device)
    alpha = torch.zeros(n, 1, device=device)
    alpha[perm[:m]] = 1.0

    h = compute_h(G_list, alpha, m, n, c_sum, device)
    
    objective_history = []
    
    # Step 3: Iterative Single-Swap Loop
    for t in range(max_iters):
        
        # Compute Objective D(alpha)
        if t % log_freq == 0:
            alpha_mask = (alpha.squeeze() == 1.0)
            
            # D(alpha) = 1/2 * alpha^T h(alpha) - 1/(mn) * alpha^T c
            term1 = 0.5 * h[alpha_mask].sum()
            term2 = (1.0 / (m * n)) * c_sum[alpha_mask].sum()
            
            current_obj = (term1 - term2).item()
            objective_history.append({"iteration": t, "objective": current_obj})

        h_in = h.clone()
        h_in[alpha == 0] = -float('inf')
        i_t = torch.argmax(h_in).item()

        h_out = h.clone()
        h_out[alpha == 1] = float('inf')
        j_t = torch.argmin(h_out).item()

        gamma_t = h[i_t].item() - h[j_t].item()

        if gamma_t <= 0:
            break

        alpha_tilde = alpha.clone()
        alpha_tilde[i_t] = 0.0
        alpha_tilde[j_t] = 1.0

        h_tilde = compute_h(G_list, alpha_tilde, m, n, c_sum, device)
        
        delta_t = 0.5 * (h_tilde[j_t] - h_tilde[i_t] + h[j_t] - h[i_t]).item()

        if delta_t < 0:
            alpha = alpha_tilde
            h = h_tilde
        else:
            break

    # Record final objective state before exiting
    alpha_mask = (alpha.squeeze() == 1.0)
    term1 = 0.5 * h[alpha_mask].sum()
    term2 = (1.0 / (m * n)) * c_sum[alpha_mask].sum()
    final_obj = (term1 - term2).item()
    objective_history.append({"iteration": t, "objective": final_obj})

    selected_indices = torch.where(alpha.squeeze() == 1.0)[0].cpu()
    print('Total number of iterations', t)
    return selected_indices, objective_history, t


def main():
    # ── 0. Argument Parsing ───────────────────────────────────────────────────────
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fraction', default=0.3, type=float)
    p.add_argument('-seed', '--seed', default=42, type=int)
    p.add_argument('--checkpoint_dir', required=True, type=str, help='Directory containing the trajectory .pth checkpoints')
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
    
    # ── 2. Logging & Artifact Setup ───────────────────────────────────────────────
    LOG_FILE = os.path.join(base_save_directory, f"kl_faithful_{budget}_training.log")
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

    MODEL_SAVE_PATH   = os.path.join(base_save_directory, f"kl_faithful_{budget}_model.pth")
    INDICES_SAVE_PATH = os.path.join(base_save_directory, f"kl_faithful_{budget}_indices.pt")
    METRICS_SAVE_PATH = os.path.join(base_save_directory, f"kl_faithful_{budget}_metrics.json")
    OBJ_SAVE_PATH     = os.path.join(base_save_directory, f"kl_faithful_{budget}_objective.json")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Running KL-Faithful selection training on device: {device}")

    trainset, valset = random_split(
        full_train, 
        [n_train, n_val], 
        generator=torch.Generator().manual_seed(args.seed)
    )

    eval_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    testloader  = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # ── 3. Feature Extraction & Algorithm 1 Selection ───────────────────────────
    reference_model = ResNet18(num_classes=100)
    reference_model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    reference_model.maxpool = nn.Identity()
    reference_model = reference_model.to(device)

    if not os.path.exists(args.checkpoint_dir):
        raise ValueError(f"Checkpoint directory {args.checkpoint_dir} does not exist.")
        
    checkpoint_paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pth")))
    if len(checkpoint_paths) == 0:
        raise ValueError(f"No .pth files found in {args.checkpoint_dir}")
        
    logger.info(f"Found {len(checkpoint_paths)} checkpoints. Extracting trajectory features...")
    G_list = load_trajectory_features(reference_model, eval_loader, checkpoint_paths, device)

    logger.info(f"Running Gradient-Ranked Single-Swap Descent (Budget: {budget}/{n_train})...")
    
    selected_indices, objective_history, total_iters = gradient_ranked_single_swap(
        G_list, budget, max_iters=5000, log_freq=5, device=device
    )
    logger.info(f"Algorithm 1 converged/finished in {total_iters} iterations.")

    # Save selection artifacts
    torch.save(selected_indices, INDICES_SAVE_PATH)
    logger.info(f"Selected {len(selected_indices)} indices and saved to {INDICES_SAVE_PATH}")
    
    with open(OBJ_SAVE_PATH, 'w') as f:
        json.dump(objective_history, f, indent=4)
    logger.info(f"Saved algorithm objective history to {OBJ_SAVE_PATH}")

    # ── 4. Retrain Model on Selected Subset ──────────────────────────────────────
    selected_subset = Subset(trainset, selected_indices)
    trainloader = DataLoader(selected_subset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)

    model = ResNet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    logger.info("Starting retraining loop on KL-Faithful subset...")
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