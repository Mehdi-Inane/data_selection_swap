"""The cost breakdown every selector's timing record carries."""

import json
import logging
import os
import tempfile
import unittest

from data_selection_swap import common


class TimingRecordTest(unittest.TestCase):
    def record(self, **extra):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, 'train_time.json'), 'w') as fh:
                json.dump({'epoch_seconds': [1.5, 2.5, 3.0]}, fh)
            timer = common.SelectionTimer('cpu')
            for phase in ('extra_training', 'scoring', 'selection'):
                with timer.phase(phase):
                    pass
            path = os.path.join(tmp, 'timing.json')
            common.save_timing(path, timer, logging.getLogger(__name__),
                               checkpoint_dir=tmp, **extra)
            with open(path) as fh:
                return json.load(fh)

    def test_reference_and_extra_training_costs(self):
        record = self.record(reference_epochs_used=2, extra_training_epochs=200)
        self.assertEqual(record['ref_epochs_used'], 2)
        self.assertAlmostEqual(record['ref_train_sec'], 4.0)  # first two epochs
        self.assertEqual(record['extra_probe_epochs'], 200)
        self.assertAlmostEqual(record['total_selection_sec'],
                               sum(record['phases_seconds'].values()))
        self.assertEqual(record['scoring_or_sketch_sec'], record['phases_seconds']['scoring'])
        self.assertEqual(record['swap_selection_sec'], record['phases_seconds']['selection'])

    def test_a_method_without_the_trajectory_costs_nothing_there(self):
        record = self.record(reference_epochs_used=0)
        self.assertEqual(record['ref_train_sec'], 0.0)
        self.assertEqual(record['extra_probe_epochs'], 0)


if __name__ == '__main__':
    unittest.main()
