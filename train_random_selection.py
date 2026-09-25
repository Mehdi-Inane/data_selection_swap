"""
train_random_selection.py
Uniform random subset under the shared protocol (same split, model,
retrain loop and output layout as every other method in this suite).
"""

import torch

import selection_common as common


def main():
    p = common.base_argparser("Random subset selection")
    args = p.parse_args()

    common.set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = common.get_dataset_config(args.dataset, args.data_dir, download=args.download)
    trainset, _, _, n_train = common.make_loaders(
        cfg, args.seed, args.batch_size, args.num_workers, args.scoring_transform)
    budget = int(args.fraction * n_train)

    paths = common.get_save_paths('random', args.dataset, budget, args.seed)
    logger = common.get_logger(__name__, paths['log'])
    logger.info(f"Dataset={args.dataset}  budget={budget}  seed={args.seed}  device={device}")

    timer = common.SelectionTimer(device)
    with timer.phase('selection'):
        selected_indices = torch.randperm(
            n_train, generator=torch.Generator().manual_seed(args.seed))[:budget]

    common.save_timing(paths['timing'], timer, logger, method='random', reference_epochs_used=0)
    common.finish(cfg, trainset, selected_indices, device, args, logger, paths)


if __name__ == "__main__":
    main()
