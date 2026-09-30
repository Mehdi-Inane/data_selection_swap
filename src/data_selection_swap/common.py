"""Shared infrastructure for the data-selection comparison suite.

The factored KL-Faithful workflow is the source of truth for the experimental
protocol. Every baseline module in :mod:`data_selection_swap.selection`
imports from here, so that dataset splits, model architecture,
optimizer/schedule, and evaluation protocol are byte-for-byte identical
across methods. The ONLY thing that is allowed to differ between
scripts is the selection rule that produces `selected_indices`.

See ``docs/experimental_protocol.md`` for the reasoning behind what is shared
and what is deliberately allowed to differ (e.g. EL2N's need for K
independently-initialised short probe runs, vs. MoSo / KL-Faithful's
shared use of the same training trajectory, vs. Herding / Moderate
Coreset / GraphCut / Facility Location's use of a single converged
embedding).
"""

from __future__ import annotations

import copy
import fcntl
import glob
import hashlib
import heapq
import json
import logging
import math
import os
import random
import re
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as tv_models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.models.googlenet import BasicConv2d

DATASETS = ('cifar10', 'cifar100', 'mnist', 'imagenet')
ARCHITECTURES = (
    'resnet18', 'vgg16', 'inception',
    # Retained for the existing mh-only controlled experiments.
    'resnet34', 'lenet', 'convnet',
)
# Backward-compatible default. New code should call ``feature_dim(model)``.
FEAT_DIM = 512


def config_fingerprint(config: dict, length: int = 12) -> str:
    """Stable identifier for an experiment's immutable configuration."""
    payload = json.dumps(config, sort_keys=True, separators=(',', ':'),
                         default=str).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:length]


def state_dict_fingerprint(state_dict) -> str:
    """Content hash of model tensors, independent of ``torch.save`` metadata."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(b'\0')
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(b'\0')
        digest.update(json.dumps(list(tensor.shape)).encode('ascii'))
        digest.update(b'\0')
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def ordered_indices_fingerprint(indices) -> str:
    """Hash an ordered split mapping in a platform-independent representation.

    Pool-relative selector indices only have meaning together with this order.
    Sorting before hashing would therefore hide exactly the split-order mismatch
    that trajectory consumers need to reject.
    """
    values = np.asarray([int(value) for value in indices], dtype='<i8')
    digest = hashlib.sha256()
    digest.update(str(values.size).encode('ascii'))
    digest.update(b'\0')
    digest.update(values.tobytes(order='C'))
    return digest.hexdigest()


def atomic_json_dump(record: dict, path: str) -> None:
    """Write JSON atomically so interrupted jobs do not leave partial files."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, 'w') as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_torch_save(record, path: str) -> None:
    """Atomically publish a torch artifact after the full payload is durable."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(record, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class LeNet(nn.Module):
    """LeNet-style classifier with a named ``linear`` last layer.

    Adaptive pooling makes the same implementation valid for MNIST's 28x28
    inputs and CIFAR's 32x32 inputs.  Exposing the final layer as ``linear``
    keeps the factored last-layer selector architecture independent.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(84, num_classes)

    def forward(self, x):
        return self.linear(self.project(self.features(x)))


class ConvNet(nn.Module):
    """Small conventional CNN used as a second non-ResNet architecture."""

    def __init__(self, in_channels: int, num_classes: int, width: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.linear = nn.Linear(2 * width, num_classes)

    def forward(self, x):
        return self.linear(self.features(x))


class TorchvisionResNet(nn.Module):
    """torchvision ResNet wrapper exposing its classifier as ``linear``."""

    def __init__(self, depth: int, in_channels: int, num_classes: int,
                 small_images: bool):
        super().__init__()
        constructor = {18: tv_models.resnet18, 34: tv_models.resnet34}[depth]
        self.backbone = _torchvision_model(constructor)
        if small_images:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()
        elif in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.linear(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configuration shared by every maintained workflow
# ─────────────────────────────────────────────────────────────────────────────
def get_dataset_config(dataset: str, data_dir: str, download: bool = False,
                       architecture: str = None) -> dict:
    """Return transforms, datasets, and model metadata for ``dataset``.

    ``architecture`` is stored in the returned configuration so every stage
    (trajectory training, selection, and subset retraining) builds the same
    network.  Leaving it unset preserves the historical ResNet-18 default for
    colour datasets and uses LeNet for MNIST.
    """
    dataset = dataset.lower()
    default_architecture = 'lenet' if dataset == 'mnist' else 'resnet18'
    architecture = architecture or default_architecture
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: {architecture!r}. Choose one of {ARCHITECTURES}."
        )
    if dataset == 'cifar10':
        # Match the adapted main-branch baseline configuration exactly.
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        tf_tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        tf_te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return dict(
            dataset='cifar10', num_classes=10, num_epochs=300,
            small_images=True, cifar_style=True, input_channels=3,
            architecture=architecture,
            full_train=datasets.CIFAR10(root=data_dir, train=True, download=download, transform=tf_tr),
            testset=datasets.CIFAR10(root=data_dir, train=False, download=download, transform=tf_te),
        )

    if dataset == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        tf_tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        tf_te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return dict(
            dataset='cifar100', num_classes=100, num_epochs=300,
            small_images=True, cifar_style=True, input_channels=3,
            architecture=architecture,
            full_train=datasets.CIFAR100(root=data_dir, train=True,  download=download, transform=tf_tr),
            testset   =datasets.CIFAR100(root=data_dir, train=False, download=download, transform=tf_te),
        )

    if dataset == 'mnist':
        mean, std = (0.1307,), (0.3081,)
        tf_tr = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        tf_te = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        return dict(
            dataset='mnist', num_classes=10, num_epochs=100,
            small_images=True, cifar_style=True, input_channels=1,
            architecture=architecture,
            full_train=datasets.MNIST(root=data_dir, train=True, download=download,
                                      transform=tf_tr),
            testset=datasets.MNIST(root=data_dir, train=False, download=download,
                                   transform=tf_te),
        )

    if dataset == 'imagenet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        tf_tr = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        tf_te = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        return dict(
            dataset='imagenet', num_classes=1000, num_epochs=350,
            small_images=False, cifar_style=False, input_channels=3,
            architecture=architecture,
            full_train=datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=tf_tr),
            testset   =datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=tf_te),
        )

    raise ValueError(
        f"Unknown dataset: {dataset!r}. Choose one of {DATASETS}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model construction, including the shared CIFAR stem
# ─────────────────────────────────────────────────────────────────────────────
def _torchvision_model(constructor, **kwargs):
    """Call a torchvision model builder on modern (weights=) and legacy APIs."""
    try:
        return constructor(weights=None, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'weights'" not in str(exc):
            raise
        return constructor(pretrained=False, **kwargs)


def build_model(cfg: dict, num_classes: int, device: str,
                architecture: str = None) -> nn.Module:
    """Build a supported network with a common ``linear`` classifier API."""
    architecture = architecture or cfg.get('architecture', 'resnet18')
    in_channels = int(cfg.get('input_channels', 3))
    small_images = bool(cfg.get('small_images', cfg.get('cifar_style', False)))

    if architecture == 'resnet18':
        # Use the raw torchvision layout so origin/main checkpoints load
        # strictly without key rewriting.
        model = _torchvision_model(tv_models.resnet18, num_classes=num_classes)
        if small_images:
            model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        elif in_channels != 3:
            model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif architecture == 'vgg16':
        model = _torchvision_model(tv_models.vgg16_bn, num_classes=num_classes)
        if in_channels != 3:
            model.features[0] = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1)
        if small_images:
            model.avgpool = nn.AdaptiveAvgPool2d(1)
            model.classifier = nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )
    elif architecture == 'inception':
        model = _torchvision_model(
            tv_models.googlenet,
            aux_logits=False,
            init_weights=True,
            num_classes=num_classes,
        )
        if small_images:
            model.conv1 = BasicConv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1)
            model.maxpool1 = nn.Identity()
        elif in_channels != 3:
            model.conv1 = BasicConv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3)
    elif architecture == 'resnet34':
        model = TorchvisionResNet(34, in_channels, num_classes, small_images)
    elif architecture == 'lenet':
        model = LeNet(in_channels, num_classes)
    elif architecture == 'convnet':
        model = ConvNet(in_channels, num_classes)
    else:
        raise ValueError(
            f"Unknown architecture: {architecture!r}. Choose one of {ARCHITECTURES}."
        )
    return model.to(device)


def classifier_layer(model: nn.Module) -> nn.Linear:
    """Return the final linear classifier used by factored-gradient code."""
    layer = getattr(model, 'linear', None)
    if isinstance(layer, nn.Linear):
        return layer
    layer = getattr(model, 'fc', None)
    if isinstance(layer, nn.Linear):
        return layer
    classifier = getattr(model, 'classifier', None)
    if isinstance(classifier, nn.Linear):
        return classifier
    if isinstance(classifier, nn.Sequential):
        for candidate in reversed(classifier):
            if isinstance(candidate, nn.Linear):
                return candidate
    raise TypeError(
        f"{type(model).__name__} has no supported final nn.Linear classifier."
    )


def classifier_parameter_names(model: nn.Module) -> tuple[str, str | None]:
    """Return state-dict names for the final classifier weight and bias."""
    target = classifier_layer(model)
    for name, module in model.named_modules():
        if module is target:
            prefix = f"{name}." if name else ""
            bias_name = f"{prefix}bias" if target.bias is not None else None
            return f"{prefix}weight", bias_name
    raise RuntimeError("Final classifier is not registered in model.named_modules().")


def feature_dim(model: nn.Module) -> int:
    """Width of the penultimate representation feeding the classifier."""
    return int(classifier_layer(model).in_features)


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders — identical split logic/seed across every method
# ─────────────────────────────────────────────────────────────────────────────
def scoring_subset(cfg: dict, trainset: Subset, scoring_transform: str = 'test') -> Subset:
    """
    The dataset view that selection methods *score* (not train on).
    'train' reproduces the original KL runs (augmented views);
    'test' scores un-augmented images, as the Moderate-DS and data_diet
    reference implementations do. Indices are identical either way.
    """
    if scoring_transform == 'train':
        return trainset
    clean = copy.copy(cfg['full_train'])
    clean.transform = cfg['testset'].transform
    return Subset(clean, trainset.indices)


def make_loaders(cfg: dict, seed: int, batch_size: int, num_workers: int,
                 scoring_transform: str = 'test'):
    n_val   = int(0.1 * len(cfg['full_train']))
    n_train = len(cfg['full_train']) - n_val
    trainset, _ = random_split(
        cfg['full_train'], [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    # shuffle=False: sample i of eval_loader always corresponds to selected_indices[i]
    eval_loader = DataLoader(scoring_subset(cfg, trainset, scoring_transform),
                             batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=num_workers)
    testloader = DataLoader(cfg['testset'], batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=num_workers)
    return trainset, eval_loader, testloader, n_train


# ─────────────────────────────────────────────────────────────────────────────
# Swap candidate enumeration
#   Every swap-based selector ranks exchanges by the linear term
#   h_j - h_i. Enumerating pairs in non-decreasing order of it is what
#   lets an exact search stop early: the curvature term is non-negative,
#   so once some pair's exact change beats the next unexamined linear
#   term, no remaining pair can do better.
# ─────────────────────────────────────────────────────────────────────────────
def ranked_pairs(drop_idx, drop_h, add_idx, add_h):
    """
    Yield (linear, i, j) over drop x add in non-decreasing add_h - drop_h.

    drop_h must be sorted descending and add_h ascending. The linear term
    is then non-decreasing in both positions, so a heap over the frontier
    yields pairs in order while holding at most O(|drop|) of them; the
    full |drop| x |add| product is never materialised. Each pair is
    pushed exactly once: (a, b>0) from (a, b-1), and (a, 0) from
    (a-1, 0).
    """
    na, nb = len(drop_h), len(add_h)
    if na == 0 or nb == 0:
        return
    heap = [(float(add_h[0] - drop_h[0]), 0, 0)]
    while heap:
        lin, a, b = heapq.heappop(heap)
        yield lin, int(drop_idx[a]), int(add_idx[b])
        if b + 1 < nb:
            heapq.heappush(heap, (float(add_h[b + 1] - drop_h[a]), a, b + 1))
        if b == 0 and a + 1 < na:
            heapq.heappush(heap, (float(add_h[0] - drop_h[a + 1]), a + 1, 0))


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────────────────────────────────────────────
def torch_load(path, *, map_location='cpu'):
    """Load a checkpoint on both legacy and modern PyTorch torch.load APIs."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError as exc:
        # Older torch rejects the kwarg; some 1.x builds accept it but fail in
        # Unpickler with "'weights_only' is an invalid keyword argument ...".
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location=map_location)


def list_checkpoints(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    ckpt_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pth")))
    if not ckpt_paths:
        raise ValueError(f"No .pth checkpoints in {checkpoint_dir}")
    return ckpt_paths


def load_model_weights(path: str, map_location='cpu'):
    """Load weights from either a bare state dict or a reference record."""
    obj = torch_load(path, map_location=map_location)
    if isinstance(obj, dict) and 'model' in obj:
        return obj['model']
    if isinstance(obj, dict) and 'state_dict' in obj:
        return obj['state_dict']
    return obj


def checkpoint_epoch(path: str) -> int:
    """Epoch number of a full-data checkpoint (checkpoint_XXX.pth)."""
    m = re.search(r'(\d+)\.pth$', os.path.basename(path))
    if m is None:
        raise ValueError(f"Cannot parse epoch from checkpoint name: {path}")
    return int(m.group(1))


def reference_epochs_used(ckpt_paths) -> int:
    """How far into the shared trajectory a method had to train (0 if none)."""
    epochs = []
    for path in ckpt_paths:
        try:
            epochs.append(checkpoint_epoch(path))
        except ValueError:
            pass
    return max(epochs, default=0)


def resolve_embedding_checkpoint(checkpoint_dir: str, override: str = None) -> str:
    """
    One-shot methods (Herding / Moderate Coreset / GraphCut / Facility
    Location) need a single reference embedding, not a full trajectory.
    Default: the LAST checkpoint in --checkpoint_dir, i.e. exactly the
    converged network the KL-Faithful / MoSo trajectory also ends on —
    this is the fairest single point of comparison, since it costs no
    extra training compute beyond what every other method already uses.
    """
    if override is not None:
        if not os.path.isfile(override):
            raise ValueError(f"--embedding_checkpoint not found: {override}")
        return override
    return list_checkpoints(checkpoint_dir)[-1]


def cosine_lr(base_lr: float, epoch: int, num_epochs: int) -> float:
    """LR used during a 1-indexed epoch by the full-data cosine schedule."""
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * (epoch - 1) / num_epochs))


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer factory
#   The single place an optimizer is built, so the reference trajectory
#   (training.reference) and the retrain-on-subset loop below cannot drift
#   apart. Scoring checkpoints from an SGD reference and then retraining
#   the selected model with Adam confounds the comparison, so the selection
#   scripts default --retrain_optimizer to whatever the objective assumed.
# ─────────────────────────────────────────────────────────────────────────────
def load_or_create_init(model: nn.Module, path, logger=None) -> str:
    """
    Pin the retraining initialisation so two selectors are compared from
    the same weights. The first caller writes the file, everyone after
    loads it; without this, each run starts from its own random draw and
    part of any accuracy gap is initialisation noise.
    """
    if not path:
        return "fresh initialisation (pass --init_ckpt to share one)"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    device = next(model.parameters()).device
    lock_path = f"{path}.lock"
    with open(lock_path, 'a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if os.path.exists(path):
            model.load_state_dict(torch_load(path, map_location=device))
            note = f"initialisation loaded from {path}"
        else:
            tmp = f"{path}.tmp.{os.getpid()}"
            torch.save(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                tmp)
            os.replace(tmp, path)
            note = f"initialisation saved to {path}"
            if logger:
                logger.info(f"Wrote a shared retraining initialisation to {path}")
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return note


def build_optimizer(params, kind: str, lr: float, weight_decay: float = 5e-4,
                    betas=(0.9, 0.999), eps: float = 1e-8):
    if kind == 'sgd':
        return optim.SGD(params, lr=lr, momentum=0.9,
                         weight_decay=weight_decay, nesterov=True)
    if kind in ('adam', 'adamw'):
        cls = optim.AdamW if kind == 'adamw' else optim.Adam
        return cls(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {kind!r}. Choose sgd, adam or adamw.")


# ─────────────────────────────────────────────────────────────────────────────
# Feature / probability extraction
#   Every baseline that needs penultimate features and/or softmax
#   probabilities reads them through this single hook-based pass, so
#   "what the model sees" is defined identically everywhere (same hook
#   point as extract_factored_features in selection.kl).
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_probs_features_targets(model: nn.Module, dataloader, device: str):
    """One forward pass -> (probs [n,C], phi [n,F], targets [n])."""
    model.eval()
    feats = {}

    def _hook(_, inp, __):
        feats['phi'] = inp[0]

    handle = classifier_layer(model).register_forward_hook(_hook)

    probs_buf, phi_buf, targets_buf = [], [], []
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs_buf.append(torch.softmax(outputs, dim=1).cpu())
        phi_buf.append(feats['phi'].cpu())
        targets_buf.append(targets.clone())

    handle.remove()
    return torch.cat(probs_buf), torch.cat(phi_buf), torch.cat(targets_buf)


def logit_grads(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """L = softmax - onehot: the logit-gradient factor used by KL-Faithful."""
    one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
    return probs - one_hot


# ─────────────────────────────────────────────────────────────────────────────
# Per-class kernels
#   Herding and GraphCut / Facility Location are pairwise methods run
#   within each class. They can consume either
#     'features'        K = Φ Φ^T                 at the embedding checkpoint
#     'grad'            K = (L L^T) ⊙ (Φ Φ^T)     at the embedding checkpoint
#     'grad_trajectory' K = Σ_t (L_t L_t^T) ⊙ (Φ_t Φ_t^T)  over every checkpoint
#   'grad_trajectory' is exactly the kernel KL-Faithful optimises over,
#   restricted to one class — the "same information" ablation.
# ─────────────────────────────────────────────────────────────────────────────
def build_class_kernels(kind: str, model: nn.Module, eval_loader, checkpoint_dir: str,
                         embedding_checkpoint: str, num_classes: int, device: str):
    """Return (kernels: list[[n_c,n_c] CPU tensor], class_idx: list[LongTensor], ckpts_used)."""
    if kind == 'grad_trajectory':
        ckpts = list_checkpoints(checkpoint_dir)
    else:
        ckpts = [resolve_embedding_checkpoint(checkpoint_dir, embedding_checkpoint)]

    kernels = class_idx = None
    for ckpt in ckpts:
        model.load_state_dict(load_model_weights(ckpt, map_location=device))
        probs, phi, targets = extract_probs_features_targets(model, eval_loader, device)
        if class_idx is None:
            class_idx = [torch.where(targets == c)[0] for c in range(num_classes)]
            kernels = [None] * num_classes
        L = logit_grads(probs, targets) if kind != 'features' else None
        for c, idx in enumerate(class_idx):
            phi_c = phi[idx].to(device)
            K = phi_c @ phi_c.T
            if L is not None:
                L_c = L[idx].to(device)
                K = K * (L_c @ L_c.T)
            K = K.cpu()
            kernels[c] = K if kernels[c] is None else kernels[c] + K
    return kernels, class_idx, ckpts


def kernel_to_similarity(K: torch.Tensor) -> torch.Tensor:
    """
    Non-negative similarity for submodular functions, following apricot's
    own convention for metric='euclidean':  S = max(D) - D, with D the
    RKHS distance D_ij = sqrt(K_ii + K_jj - 2 K_ij).
    """
    diag = torch.diagonal(K)
    D = (diag.unsqueeze(0) + diag.unsqueeze(1) - 2.0 * K).clamp_min(0.0).sqrt()
    return D.max() - D


# ─────────────────────────────────────────────────────────────────────────────
# Per-class budget helper
#   Used by methods that are natively defined per class (Herding,
#   MoSo, GraphCut / Facility Location), so every class contributes
#   budget * (class_count / n_train) samples, matching the public
#   reference implementations linked for each method. Uses
#   largest-remainder rounding so quotas sum exactly to `budget`.
# ─────────────────────────────────────────────────────────────────────────────
def per_class_budget(targets: torch.Tensor, num_classes: int, budget: int) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=num_classes).float()
    raw    = counts * (budget / counts.sum())
    quota  = raw.floor().long()
    remainder = int(budget - quota.sum().item())
    if remainder > 0:
        frac_order = torch.argsort(raw - quota.float(), descending=True)
        for c in frac_order[:remainder]:
            quota[c] += 1
    return quota  # LongTensor [num_classes]


def per_class_topk(scores: torch.Tensor, targets: torch.Tensor, num_classes: int,
                   budget: int) -> torch.Tensor:
    quota = per_class_budget(targets, num_classes, budget)
    selected = []
    for c in range(num_classes):
        idx = torch.where(targets == c)[0]
        q = min(int(quota[c].item()), idx.numel())
        if q > 0:
            selected.append(idx[torch.topk(scores[idx], q).indices])
    return torch.cat(selected)


# ─────────────────────────────────────────────────────────────────────────────
# Short full-data training (EL2N probes, MoSo surrogates)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        criterion(model(inputs), targets).backward()
        optimizer.step()


# ─────────────────────────────────────────────────────────────────────────────
# Selection-time accounting
#   Wall-clock, CUDA-synchronised, split into phases:
#     extra_training : training the method needs beyond the shared
#                      full-data trajectory (EL2N probes, MoSo surrogates)
#     scoring        : checkpoint loading + forward/backward passes over
#                      the training set to build features/scores/kernels
#     selection      : the combinatorial rule that turns scores into indices
#   `reference_epochs_used` records how much of the shared trajectory the
#   method consumed, so analysis.comparison can add the amortised cost of
#   those epochs from the full-data trainer's timing file.
# ─────────────────────────────────────────────────────────────────────────────
class SelectionTimer:
    def __init__(self, device: str):
        self.device = device
        self.phases = {}

    def _sync(self):
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def phase(self, name: str):
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.phases[name] = self.phases.get(name, 0.0) + time.perf_counter() - t0

    def total(self) -> float:
        return sum(self.phases.values())


def save_timing(path: str, timer: SelectionTimer, logger, **extra):
    record = dict(
        phases_seconds=timer.phases,
        selection_seconds=timer.total(),
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
        **extra,
    )
    atomic_json_dump(record, path)
    logger.info(f"Selection time: {timer.total():.1f}s  phases="
                + ", ".join(f"{k}={v:.1f}s" for k, v in timer.phases.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Save-path / logging convention (one line change — method name — vs.
# selection.kl's own base_dir / logging setup)
# ─────────────────────────────────────────────────────────────────────────────
def get_save_paths(method: str, dataset: str, architecture: str,
                   budget: int, seed: int) -> dict:
    scratch_path = os.environ.get("SCRATCH", "/home/mila/a/ahmedm/scratch")
    base_dir = (
        f"{scratch_path}/data_selection_baselines/{method}/{dataset}/"
        f"{architecture}/{budget}/seed_{seed}"
    )
    os.makedirs(base_dir, exist_ok=True)
    return dict(
        base_dir=base_dir,
        model  =os.path.join(base_dir, f"{method}_{budget}_model.pth"),
        indices=os.path.join(base_dir, f"{method}_{budget}_indices.pt"),
        metrics=os.path.join(base_dir, f"{method}_{budget}_metrics.json"),
        scores =os.path.join(base_dir, f"{method}_{budget}_scores.json"),
        timing =os.path.join(base_dir, f"{method}_{budget}_timing.json"),
        log    =os.path.join(base_dir, f"{method}_{budget}_training.log"),
    )


def get_logger(name: str, log_path: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(name)


# ─────────────────────────────────────────────────────────────────────────────
# Shared argparser — every baseline adds only its method-specific flags
# ─────────────────────────────────────────────────────────────────────────────
def base_argparser(description: str):
    import argparse

    def open_unit_interval(value):
        parsed = float(value)
        if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
            raise argparse.ArgumentTypeError('fraction must be finite and strictly between 0 and 1')
        return parsed

    p = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fraction',       default=0.3,         type=open_unit_interval)
    p.add_argument('--seed', '-seed',  default=42,          type=int)
    p.add_argument('--checkpoint_dir', required=True,       type=str,
                   help='Shared reference-trajectory directory')
    p.add_argument('--dataset',        default='cifar100',  choices=list(DATASETS))
    p.add_argument('--architecture',   default=None, choices=list(ARCHITECTURES),
                   help='Network architecture. Defaults to resnet18, or lenet for MNIST.')
    p.add_argument('--data_dir',       required=True,       type=str)
    p.add_argument('--batch_size',     default=128,         type=int)
    p.add_argument('--lr',             default=0.01,        type=float,
                   help='Learning rate for the final retrain-on-subset stage')
    p.add_argument('--retrain_optimizer', default='sgd', choices=['sgd', 'adam', 'adamw'],
                   help='Optimizer used only for the controlled final subset retrain.')
    p.add_argument('--weight_decay',   default=5e-4,        type=float)
    p.add_argument('--beta1',          default=0.9,         type=float)
    p.add_argument('--beta2',          default=0.999,       type=float)
    p.add_argument('--adam_eps',       default=1e-8,        type=float)
    p.add_argument('--init_ckpt',      default=None,        type=str,
                   help='Shared initialization file used by every compared method.')
    p.add_argument('--num_workers',    default=4,           type=int)
    p.add_argument('--download',       action='store_true', default=False)
    p.add_argument('--scoring_transform', default='test', choices=['train', 'test'],
                   help="Transform applied to the training set when scoring it. "
                        "'train' matches the original KL runs; use the same value "
                        "for every method in a comparison.")
    p.add_argument('--selection_only', action='store_true', default=False,
                   help='Stop after saving indices + timing (skip retraining).')
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Shared retrain-on-subset loop (identical to steps 8-9 of
# selection.kl) — the only stage that consumes GPU-hours
# proportional to the *final* budget, run identically for every method
# so that accuracy differences trace back to `selected_indices` alone.
# ─────────────────────────────────────────────────────────────────────────────
def retrain_on_subset(cfg, trainset, selected_indices, device, args, logger, paths):
    train_generator = torch.Generator().manual_seed(int(args.seed) + 10_000)
    trainloader = DataLoader(
        Subset(trainset, selected_indices),
        batch_size=args.batch_size, shuffle=True, pin_memory=True,
        num_workers=args.num_workers, generator=train_generator,
    )
    testloader = DataLoader(cfg['testset'], batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=args.num_workers)

    # Selection methods consume different random streams. Reset immediately
    # before construction and use a dedicated loader generator so final
    # accuracy differs because of the subset, not method-specific RNG use.
    set_seed(args.seed)
    model = build_model(cfg, cfg['num_classes'], device)
    init_note = load_or_create_init(model, getattr(args, 'init_ckpt', None), logger)
    criterion = nn.CrossEntropyLoss()
    kind = getattr(args, 'retrain_optimizer', 'sgd')
    optimizer = build_optimizer(model.parameters(), kind, args.lr,
                                weight_decay=getattr(args, 'weight_decay', 5e-4),
                                betas=(getattr(args, 'beta1', 0.9),
                                       getattr(args, 'beta2', 0.999)),
                                eps=getattr(args, 'adam_eps', 1e-8))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'])

    logger.info(f"Retraining on subset of size {len(selected_indices)} "
                f"with {kind} (lr={args.lr}), {init_note} ...")
    accuracy_history = {}

    for epoch in range(1, cfg['num_epochs'] + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        if epoch % 50 == 0 or epoch == cfg['num_epochs']:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in testloader:
                    x, y = x.to(device), y.to(device)
                    correct += model(x).argmax(1).eq(y).sum().item()
                    total   += y.size(0)
            acc = 100.0 * correct / total
            accuracy_history[epoch] = acc
            logger.info(f"Epoch {epoch:03d}/{cfg['num_epochs']} | "
                        f"Test Acc: {acc:.2f}% | "
                        f"Train Loss: {running_loss / len(trainloader):.4f}")

    torch.save(model.state_dict(), paths['model'])
    with open(paths['metrics'], 'w') as fh:
        json.dump(accuracy_history, fh, indent=4)
    logger.info(f"Saved all artefacts to {paths['base_dir']}")
    return accuracy_history


def finish(cfg, trainset, selected_indices, device, args, logger, paths):
    """Save indices, then retrain unless --selection_only."""
    torch.save(selected_indices, paths['indices'])
    if args.selection_only:
        logger.info("--selection_only set: skipping retraining.")
        return
    retrain_on_subset(cfg, trainset, selected_indices, device, args, logger, paths)
