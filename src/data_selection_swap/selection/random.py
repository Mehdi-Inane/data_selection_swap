"""Uniform random subset under the shared protocol (same pool, model,
retrain loop and output layout as every other method in this suite).

``--stratified`` draws at the per-class quotas instead
(``subsets.initial_subset`` with labels), which is exactly the start the
class-balanced second-order swap uses at the same seed.
"""

import torch

from .. import common, subsets


def main():
    p = common.base_argparser("Random subset selection")
    p.add_argument('--stratified', action='store_true', default=False,
                   help='Draw at the per-class quotas of subsets.per_class_budget.')
    args = p.parse_args()
    method = 'random_stratified' if args.stratified else 'random'

    common.set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = common.get_dataset_config(args.dataset, args.data_dir, download=args.download,
                                    architecture=args.architecture)
    trainset, _, _, n_train = common.make_loaders(
        cfg, args.seed, args.batch_size, args.num_workers, args.scoring_transform)
    budget = int(args.fraction * n_train)

    paths = common.get_save_paths(
        method, args.dataset, cfg['architecture'], budget, args.seed,
        variant=common.storage_variant(args.retrain_optimizer))
    logger = common.get_logger(__name__, paths['log'])
    logger.info(f"Dataset={args.dataset}  budget={budget}  seed={args.seed}  device={device}")

    timer = common.SelectionTimer(device)
    with timer.phase('selection'):
        if args.stratified:
            labels = torch.as_tensor(cfg['full_train'].targets)[trainset.indices]
            selected_indices = subsets.initial_subset(
                n_train, budget, seed=args.seed, labels=labels,
                num_classes=cfg['num_classes'])
        else:
            selected_indices = torch.randperm(
                n_train, generator=torch.Generator().manual_seed(args.seed))[:budget]

    common.save_timing(paths['timing'], timer, logger, method=method, reference_epochs_used=0)
    common.finish(cfg, trainset, selected_indices, device, args, logger, paths)


if __name__ == "__main__":
    main()
