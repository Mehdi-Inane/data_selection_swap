"""Adam results live under an adam/ subfolder; SGD paths stay put."""

import os
import tempfile
import unittest

from data_selection_swap.analysis.comparison import discover_methods, run_dir
from data_selection_swap.paths import default_results_root, normalize_variant, storage_variant, tree


class ResultLayoutTest(unittest.TestCase):
    def test_sgd_paths_have_no_extra_component(self):
        self.assertEqual(normalize_variant('sgd'), '')
        self.assertEqual(storage_variant('sgd', 'uniform'), '')
        self.assertEqual(tree('/scratch', 'gradmatch_swap', 'sgd'),
                         os.path.join('/scratch', 'gradmatch_swap'))

    def test_adam_family_shares_one_subfolder(self):
        for name in ('adam', 'adamw', 'adam_update'):
            self.assertEqual(storage_variant(name), 'adam')
            self.assertEqual(
                tree('/scratch', 'data_selection_baselines', name),
                os.path.join('/scratch', 'data_selection_baselines', 'adam'))

    def test_results_root_order(self):
        saved = {name: os.environ.get(name) for name in ('RESULTS_ROOT', 'SCRATCH')}
        try:
            os.environ.pop('RESULTS_ROOT', None)
            os.environ.pop('SCRATCH', None)
            self.assertEqual(
                default_results_root(), os.path.join(os.getcwd(), 'results'))
            os.environ['SCRATCH'] = '/cluster/scratch'
            self.assertEqual(default_results_root(), '/cluster/scratch')
            os.environ['RESULTS_ROOT'] = '/explicit/results'
            self.assertEqual(default_results_root(), '/explicit/results')
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_discovery_keeps_the_two_trees_apart(self):
        with tempfile.TemporaryDirectory() as scratch:
            sgd = run_dir(scratch, 'random', 'cifar10', 'resnet18', 4500, 42)
            adam = run_dir(scratch, 'random', 'cifar10', 'resnet18', 4500, 42,
                           variant='adam')
            self.assertNotEqual(sgd, adam)
            self.assertNotIn(f'{os.sep}adam{os.sep}', sgd)
            os.makedirs(adam)
            self.assertEqual(
                discover_methods(scratch, 'cifar10', 'resnet18'),
                ['kl_faithful'])
            self.assertIn(
                'random',
                discover_methods(scratch, 'cifar10', 'resnet18', variant='adam'))

if __name__ == '__main__':
    unittest.main()
