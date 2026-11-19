"""The fine-tuning pieces: the 224-px ResNet-18, an epoch-0 checkpoint in the
reference trajectory, and the step-matched retrain."""

import argparse
import json
import logging
import math
import os
import tempfile
import unittest

import torch
import torchvision.models as tv_models

from data_selection_swap import common, trajectory


class UpsampledResNet18Test(unittest.TestCase):
    def test_torchvision_layout_so_the_imagenet_weights_load(self):
        model = common.UpsampledResNet18()
        reference = tv_models.resnet18(weights=None).state_dict()
        self.assertEqual({k: v.shape for k, v in model.state_dict().items()},
                         {k: v.shape for k, v in reference.items()})
        model.load_state_dict(reference)

    def test_runs_cifar_images_at_224_px(self):
        model = common.UpsampledResNet18(num_classes=7).eval()
        seen = {}
        model.conv1.register_forward_hook(lambda m, inp, out: seen.update(shape=inp[0].shape))
        self.assertEqual(tuple(model(torch.randn(2, 3, 32, 32)).shape), (2, 7))
        self.assertEqual(tuple(seen['shape']), (2, 3, 224, 224))


class InitialCheckpointTest(unittest.TestCase):
    """A 10-epoch trajectory (391 steps per epoch) that starts with checkpoint_000."""

    def resolve(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            lrs = [common.cosine_lr(0.01, e, 10) for e in range(1, 11)]
            records = [{'file': 'checkpoint_000.pth', 'step': 1, 'epoch': 0, 'lr': lrs[0]}]
            records += [{'file': f'checkpoint_{e:03d}.pth', 'step': 391 * e, 'epoch': e,
                         'lr': lrs[e - 1]} for e in range(1, 11)]
            for record in records:
                torch.save({}, os.path.join(tmp, record['file']))
            with open(os.path.join(tmp, trajectory.MANIFEST_NAME), 'w') as fh:
                json.dump({'steps_per_epoch': 391, 'epoch_lrs': lrs, 'num_epochs': 10,
                           'batch_size': 128, 'optimizer': 'sgd',
                           'checkpoints': records}, fh)
            checkpoints, _, _ = trajectory.resolve(tmp, optimizer='sgd', **kwargs)
            return [(c.epoch, c.step) for c in checkpoints]

    def test_the_starting_point_alone(self):
        self.assertEqual(self.resolve(epochs=[0]), [(0, 1)])

    def test_four_spread_checkpoints_are_the_configs_epochs(self):
        # configs/revision/cifar100_finetune.yaml lists them explicitly.
        self.assertEqual([e for e, _ in self.resolve(num_checkpoints=4)], [0, 3, 7, 10])


class _Pool(torch.utils.data.Dataset):
    def __init__(self, n):
        self.x = torch.randn(n, 3, 8, 8)
        self.targets = (torch.arange(n) % 3).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


class StepScheduleTest(unittest.TestCase):
    def test_subset_gets_the_references_steps_and_learning_rates(self):
        cfg = dict(num_classes=3, num_epochs=4, input_channels=3, small_images=True,
                   architecture='convnet', testset=_Pool(6))
        trainset = torch.utils.data.Subset(_Pool(50), list(range(50)))
        args = argparse.Namespace(seed=0, batch_size=8, num_workers=0, lr=0.1,
                                  retrain_schedule='steps', retrain_optimizer='sgd')
        lrs = []
        build = common.build_optimizer

        def recording_optimizer(*a, **k):
            optimizer = build(*a, **k)
            step = optimizer.step
            optimizer.step = lambda: (lrs.append(optimizer.param_groups[0]['lr']), step())
            return optimizer

        common.build_optimizer = recording_optimizer
        try:
            with tempfile.TemporaryDirectory() as tmp:
                paths = dict(base_dir=tmp, model=os.path.join(tmp, 'm.pth'),
                             metrics=os.path.join(tmp, 'metrics.json'))
                history = common.retrain_on_subset(cfg, trainset, [0, 1, 2, 3, 4], 'cpu',
                                                   args, logging.getLogger(__name__), paths)
        finally:
            common.build_optimizer = build

        steps_per_epoch = math.ceil(50 / 8)
        self.assertEqual(len(lrs), 4 * steps_per_epoch)
        self.assertEqual(lrs, [common.cosine_lr(0.1, s // steps_per_epoch + 1, 4)
                               for s in range(len(lrs))])
        self.assertEqual(list(history), [4])


if __name__ == '__main__':
    unittest.main()
