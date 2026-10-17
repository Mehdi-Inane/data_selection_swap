import importlib.util
import json
import os
import tempfile
import unittest

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_by_path(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO_ROOT, relpath))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Use the real torchvision when it is available; only then do the
# trajectory-metadata tests' stubs matter (loading them always would mask the
# real package, which common.py imports submodules from).
try:
    import torchvision.models.googlenet  # noqa: F401
except ImportError:  # pragma: no cover - headless/CI environments
    _load_by_path('test_trajectory_metadata',
                  os.path.join('tests', 'test_trajectory_metadata.py'))

from data_selection_swap import common, trajectory  # noqa: E402

migrate = _load_by_path(
    'migrate_pool_provenance',
    os.path.join('scripts', 'analyze', 'migrate_pool_provenance.py'))


def _manifest(n, seed, pool_hash, *, batch_size=10, with_immutable=True):
    steps_per_epoch = -(-n // batch_size)
    manifest = {
        'schema_version': 2, 'optimizer': 'sgd', 'lr': 0.1, 'weight_decay': 5e-4,
        'dataset': 'cifar10', 'architecture': 'resnet18', 'seed': seed,
        'num_epochs': 1, 'batch_size': batch_size, 'n_train': n,
        'steps_per_epoch': steps_per_epoch, 'pool_indices_sha256': pool_hash,
        'total_steps': steps_per_epoch, 'epochs_completed': 1,
        'epoch_lrs': [0.1],
        'checkpoints': [{'file': 'checkpoint_001.pth', 'step': steps_per_epoch,
                         'epoch': 1, 'lr': 0.1}],
        'initial_state_sha256': 'deadbeef',
    }
    if with_immutable:
        immutable = {k: manifest[k] for k in
                     ('schema_version', 'dataset', 'architecture', 'seed', 'optimizer',
                      'lr', 'weight_decay', 'num_epochs', 'batch_size', 'n_train',
                      'steps_per_epoch', 'pool_indices_sha256', 'initial_state_sha256')}
        manifest['immutable_config'] = immutable
        manifest['run_id'] = common.config_fingerprint(immutable)
    return manifest


class PoolProvenanceMigrationTests(unittest.TestCase):
    N = 97  # coprime with typical batch sizes: the perm differs from range(n)

    def _write(self, root, manifest, *, checkpoint=True):
        if checkpoint:
            open(os.path.join(root, 'checkpoint_001.pth'), 'wb').close()
        path = os.path.join(root, trajectory.MANIFEST_NAME)
        with open(path, 'w') as fh:
            json.dump(manifest, fh)
        return path

    def test_classification_of_the_two_known_pools(self):
        legacy = migrate.legacy_hash(self.N, 42)
        identity = migrate.identity_hash(self.N)
        self.assertNotEqual(legacy, identity)
        self.assertEqual(
            migrate.classify_manifest({'pool_indices_sha256': legacy}, self.N, 42),
            'legacy')
        self.assertEqual(
            migrate.classify_manifest({'pool_indices_sha256': identity}, self.N, 42),
            'identity')
        # The seed fixes the permutation: another seed's legacy hash is 'other'.
        self.assertEqual(
            migrate.classify_manifest(
                {'pool_indices_sha256': migrate.legacy_hash(self.N, 43)}, self.N, 42),
            'other')

    def test_verified_legacy_manifest_is_relabeled_and_resolve_accepts_it(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write(root, _manifest(self.N, 42, migrate.legacy_hash(self.N, 42)))
            before = json.load(open(path))

            status, _ = migrate.migrate_manifest(path, apply=True)
            self.assertEqual(status, 'verified-legacy')

            after = json.load(open(path))
            new_hash = migrate.identity_hash(self.N)
            self.assertEqual(after['pool_indices_sha256'], new_hash)
            self.assertEqual(after['immutable_config']['pool_indices_sha256'], new_hash)
            self.assertEqual(after['run_id'],
                             common.config_fingerprint(after['immutable_config']))
            self.assertNotEqual(after['run_id'], before['run_id'])
            record = after['pool_provenance_migration']
            self.assertEqual(record['old_pool_indices_sha256'],
                             before['pool_indices_sha256'])
            self.assertEqual(record['old_run_id'], before['run_id'])
            self.assertTrue(os.path.isfile(os.path.join(root, migrate.BACKUP_NAME)))

            # The whole point: resolve() now passes the identity-pool gate.
            checkpoints, _, _ = trajectory.resolve(
                root, optimizer='uniform', num_checkpoints=1,
                n_train=self.N, batch_size=10, num_epochs=1,
                expected_dataset='cifar10', expected_seed=42,
                expected_architecture='resnet18',
                expected_pool_indices_sha256=new_hash)
            self.assertEqual(len(checkpoints), 1)

    def test_dry_run_does_not_touch_the_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write(root, _manifest(self.N, 42, migrate.legacy_hash(self.N, 42)))
            before = open(path).read()
            status, _ = migrate.migrate_manifest(path, apply=False)
            self.assertEqual(status, 'verified-legacy')
            self.assertEqual(open(path).read(), before)
            self.assertFalse(os.path.exists(os.path.join(root, migrate.BACKUP_NAME)))

    def test_migration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write(root, _manifest(self.N, 42, migrate.legacy_hash(self.N, 42)))
            migrate.migrate_manifest(path, apply=True)
            after_first = open(path).read()
            status, _ = migrate.migrate_manifest(path, apply=True)
            self.assertEqual(status, 'already-migrated')
            self.assertEqual(open(path).read(), after_first)

    def test_unverifiable_manifest_is_refused_and_untouched(self):
        tampered = _manifest(self.N, 42, 'not-a-real-pool-hash')
        with tempfile.TemporaryDirectory() as root:
            path = self._write(root, tampered)
            before = open(path).read()
            status, message = migrate.migrate_manifest(path, apply=True)
            self.assertEqual(status, 'unverifiable')
            self.assertIn('neither', message)
            self.assertEqual(open(path).read(), before)
            self.assertFalse(os.path.exists(os.path.join(root, migrate.BACKUP_NAME)))

    def test_manifest_with_inconsistent_run_id_is_refused(self):
        # An edited manifest whose run_id no longer fingerprints its
        # immutable_config is not trustworthy, even if the pool hash verifies.
        manifest = _manifest(self.N, 42, migrate.legacy_hash(self.N, 42))
        manifest['run_id'] = 'ffffffffffff'
        with tempfile.TemporaryDirectory() as root:
            path = self._write(root, manifest)
            before = open(path).read()
            status, message = migrate.migrate_manifest(path, apply=True)
            self.assertEqual(status, 'unverifiable')
            self.assertIn('run_id', message)
            self.assertEqual(open(path).read(), before)

    def test_backup_is_only_written_once(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write(root, _manifest(self.N, 42, migrate.legacy_hash(self.N, 42)))
            migrate.migrate_manifest(path, apply=True)
            backup = os.path.join(root, migrate.BACKUP_NAME)
            original = open(backup).read()
            # A hypothetical second migration round must not clobber the backup.
            manifest = _manifest(self.N, 42, migrate.legacy_hash(self.N, 42))
            self._write(root, manifest)
            migrate.migrate_manifest(path, apply=True)
            self.assertEqual(open(backup).read(), original)


if __name__ == '__main__':
    unittest.main()
