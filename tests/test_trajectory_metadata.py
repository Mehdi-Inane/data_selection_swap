import json
import os
import sys
import tempfile
import types
import unittest

import torch


def _stub_optional_vision_dependencies():
    """The unit tests exercise metadata math and do not need torchvision/CORDS."""
    if 'torchvision' not in sys.modules:
        torchvision = types.ModuleType('torchvision')
        torchvision.datasets = types.ModuleType('torchvision.datasets')
        torchvision.models = types.ModuleType('torchvision.models')
        torchvision.transforms = types.ModuleType('torchvision.transforms')
        sys.modules['torchvision'] = torchvision
        sys.modules['torchvision.datasets'] = torchvision.datasets
        sys.modules['torchvision.models'] = torchvision.models
        sys.modules['torchvision.transforms'] = torchvision.transforms

    try:
        __import__('cords.utils.models')
    except ImportError:
        cords = types.ModuleType('cords')
        utils = types.ModuleType('cords.utils')
        models = types.ModuleType('cords.utils.models')

        class _UnusedResNet(torch.nn.Module):
            pass

        models.ResNet18 = _UnusedResNet
        cords.utils = utils
        utils.models = models
        sys.modules['cords'] = cords
        sys.modules['cords.utils'] = utils
        sys.modules['cords.utils.models'] = models


_stub_optional_vision_dependencies()

from data_selection_swap import common  # noqa: E402
from data_selection_swap import trajectory  # noqa: E402


class TrajectoryMetadataTests(unittest.TestCase):
    def test_spread_is_uniform_in_training_step_not_file_position(self):
        epochs = list(range(1, 11)) + list(range(20, 301, 10))
        available = [
            (f'checkpoint_{epoch:03d}.pth', epoch * 100, epoch, 0.1)
            for epoch in epochs
        ]
        picked = trajectory._pick(available, 5, 'spread', None, None)
        picked_epochs = [row[2] for row in picked]
        self.assertEqual(picked_epochs[0], 1)
        self.assertEqual(picked_epochs[-1], 300)
        self.assertGreaterEqual(picked_epochs[1], 70)
        self.assertGreaterEqual(picked_epochs[2], 140)
        self.assertGreaterEqual(picked_epochs[3], 210)

    def test_manifest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = os.path.join(root, 'checkpoint_001.pth')
            open(checkpoint, 'wb').close()
            manifest = {
                'optimizer': 'sgd',
                'dataset': 'cifar10',
                'architecture': 'resnet18',
                'seed': 42,
                'num_epochs': 1,
                'batch_size': 10,
                'n_train': 90,
                'steps_per_epoch': 9,
                'epoch_lrs': [0.1],
                'checkpoints': [
                    {'file': 'checkpoint_001.pth', 'step': 9,
                     'epoch': 1, 'lr': 0.1},
                ],
            }
            with open(os.path.join(root, trajectory.MANIFEST_NAME), 'w') as fh:
                json.dump(manifest, fh)

            with self.assertRaisesRegex(ValueError, 'seed'):
                trajectory.resolve(
                    root, optimizer='uniform', num_checkpoints=1,
                    n_train=90, batch_size=10, num_epochs=1,
                    expected_dataset='cifar10', expected_seed=43,
                    expected_architecture='resnet18')

    def test_config_fingerprint_is_order_independent(self):
        self.assertEqual(
            common.config_fingerprint({'b': 2, 'a': 1}),
            common.config_fingerprint({'a': 1, 'b': 2}),
        )
        self.assertNotEqual(
            common.config_fingerprint({'seed': 42}),
            common.config_fingerprint({'seed': 43}),
        )

    def test_ordered_pool_hash_and_manifest_guard_detect_order_changes(self):
        first = common.ordered_indices_fingerprint([4, 1, 8, 2])
        reordered = common.ordered_indices_fingerprint([1, 4, 8, 2])
        self.assertNotEqual(first, reordered)

        with tempfile.TemporaryDirectory() as root:
            checkpoint = os.path.join(root, 'checkpoint_001.pth')
            open(checkpoint, 'wb').close()
            manifest = {
                'optimizer': 'sgd', 'dataset': 'cifar10',
                'architecture': 'resnet18', 'seed': 42,
                'num_epochs': 1, 'batch_size': 10, 'n_train': 4,
                'pool_indices_sha256': first,
                'steps_per_epoch': 1, 'epoch_lrs': [0.1],
                'checkpoints': [
                    {'file': 'checkpoint_001.pth', 'step': 1,
                     'epoch': 1, 'lr': 0.1},
                ],
            }
            with open(os.path.join(root, trajectory.MANIFEST_NAME), 'w') as fh:
                json.dump(manifest, fh)

            with self.assertRaisesRegex(ValueError, 'pool_indices_sha256'):
                trajectory.resolve(
                    root, optimizer='uniform', num_checkpoints=1,
                    n_train=4, batch_size=10, num_epochs=1,
                    expected_dataset='cifar10', expected_seed=42,
                    expected_architecture='resnet18',
                    expected_pool_indices_sha256=reordered)

    def test_state_dict_fingerprint_tracks_tensor_content_not_key_order(self):
        first = {'weight': torch.arange(6.0).reshape(2, 3), 'step': torch.tensor(1)}
        reordered = {'step': torch.tensor(1), 'weight': first['weight'].clone()}
        changed = {'weight': first['weight'].clone(), 'step': torch.tensor(2)}
        self.assertEqual(
            common.state_dict_fingerprint(first),
            common.state_dict_fingerprint(reordered),
        )
        self.assertNotEqual(
            common.state_dict_fingerprint(first),
            common.state_dict_fingerprint(changed),
        )


    @staticmethod
    def _write_trajectory(root, *, pool_hash, n_train, batch_size=10):
        """One-checkpoint SGD manifest, enough for ``resolve`` to run end to end."""
        open(os.path.join(root, 'checkpoint_001.pth'), 'wb').close()
        steps_per_epoch = -(-n_train // batch_size)
        manifest = {
            'optimizer': 'sgd', 'dataset': 'cifar10', 'architecture': 'resnet18',
            'seed': 42, 'num_epochs': 1, 'batch_size': batch_size,
            'n_train': n_train, 'steps_per_epoch': steps_per_epoch,
            'epoch_lrs': [0.1], 'pool_indices_sha256': pool_hash,
            'checkpoints': [{'file': 'checkpoint_001.pth',
                             'step': steps_per_epoch, 'epoch': 1, 'lr': 0.1}],
        }
        with open(os.path.join(root, trajectory.MANIFEST_NAME), 'w') as fh:
            json.dump(manifest, fh)

    def test_full_pool_hash_rejects_a_trajectory_from_the_old_90_percent_split(self):
        # The old pool was a 45,000-image random_split; the pool is now the
        # whole training set. A trajectory carrying the old hash must be refused
        # rather than scored against the wrong pool.
        full_pool = common.ordered_indices_fingerprint(range(8))
        old_split = common.ordered_indices_fingerprint([0, 1, 2, 4, 5, 6, 7])
        self.assertNotEqual(full_pool, old_split)
        with tempfile.TemporaryDirectory() as root:
            self._write_trajectory(root, pool_hash=old_split, n_train=7)
            with self.assertRaisesRegex(ValueError, 'pool_indices_sha256'):
                trajectory.resolve(
                    root, optimizer='uniform', num_checkpoints=1,
                    n_train=8, batch_size=10, num_epochs=1,
                    expected_dataset='cifar10', expected_seed=42,
                    expected_architecture='resnet18',
                    expected_pool_indices_sha256=full_pool)

    def test_resolve_checks_the_training_batch_size_not_the_scoring_one(self):
        # score_subsets scores with batch 256 but the reference was trained with
        # 128; the trajectory check must be driven by the latter.
        pool = common.ordered_indices_fingerprint(range(8))
        with tempfile.TemporaryDirectory() as root:
            self._write_trajectory(root, pool_hash=pool, n_train=8, batch_size=128)
            kwargs = dict(optimizer='uniform', num_checkpoints=1, n_train=8,
                          num_epochs=1, expected_dataset='cifar10',
                          expected_seed=42, expected_architecture='resnet18',
                          expected_pool_indices_sha256=pool)
            trajectory.resolve(root, batch_size=128, **kwargs)
            with self.assertRaisesRegex(ValueError, 'batch_size'):
                trajectory.resolve(root, batch_size=256, **kwargs)


class LocateSeedDirTests(unittest.TestCase):
    def test_a_seed_template_is_filled_in(self):
        self.assertEqual(
            trajectory.locate_seed_dir('/a/checkpoints/seed_{seed}', 43),
            '/a/checkpoints/seed_43')

    def test_a_seedless_directory_descends_to_the_seed_child(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, 'seed_42'))
            self.assertEqual(trajectory.locate_seed_dir(root + os.sep, 42),
                             os.path.join(root + os.sep, 'seed_42'))

    def test_the_seed_child_beats_a_stale_parent_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, trajectory.MANIFEST_NAME), 'w') as fh:
                fh.write('{}')
            os.makedirs(os.path.join(root, 'seed_42'))
            self.assertEqual(trajectory.locate_seed_dir(root, 42),
                             os.path.join(root, 'seed_42'))

    def test_a_lone_parent_manifest_is_still_used_when_there_is_no_seed_child(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, trajectory.MANIFEST_NAME), 'w') as fh:
                fh.write('{}')
            self.assertEqual(trajectory.locate_seed_dir(root, 42), root)


if __name__ == '__main__':
    unittest.main()
