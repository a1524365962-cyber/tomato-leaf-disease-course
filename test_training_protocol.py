"""Standard-library tests for the completed-run protocol guard (no torch)."""
import copy
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import training_protocol
from training_protocol import (
    PROTOCOL_FIELDS,
    ProtocolMismatchError,
    build_training_protocol,
    validate_completed_run_config,
)


ROOT = Path(__file__).resolve().parent


class TrainingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        manifest = ROOT / 'manifest.json'
        cls.manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        cls.classes = sorted({row['class_name'] for row in json.loads(manifest.read_text())['rows']})
        cls.formal_paths = sorted((ROOT / 'runs').glob('*/config.json'))

    def expected(self, method: str = 'cnn', seed: int = 2026) -> dict[str, object]:
        return build_training_protocol(
            method=method,
            seed=seed,
            epochs=10,
            classes=self.classes,
            parameters=94375 if method == 'cnn' else 11180103,
            manifest_sha256=self.manifest_sha256,
        )

    def test_all_nine_formal_configs_are_accepted_unchanged(self) -> None:
        self.assertEqual(len(self.formal_paths), 9)
        expected_runs = {(method, seed) for method in ('cnn', 'resnet18', 'resnet18_aug') for seed in (2026, 2027, 2028)}
        actual_runs = set()
        for path in self.formal_paths:
            method, seed_text = path.parent.name.rsplit('_seed', 1)
            seed = int(seed_text)
            actual_runs.add((method, seed))
            with self.subTest(run=path.parent.name):
                actual = json.loads(path.read_text())
                expected = self.expected(method, seed)
                self.assertEqual(set(actual), set(expected))
                self.assertEqual(actual, expected)
                validate_completed_run_config(actual, expected, run_name=path.parent.name)
        self.assertEqual(actual_runs, expected_runs)

    def test_each_protocol_field_change_is_rejected(self) -> None:
        expected = self.expected()
        changes = {
            'method': 'resnet18',
            'seed': 2027,
            'epochs': 11,
            'batch_size': 32,
            'input_size': 224,
            'lr': 0.0003,
            'weight_decay': 0.001,
            'optimizer': 'SGD',
            'scheduler': 'StepLR',
            'pretrained': True,
            'classes': list(reversed(self.classes)),
            'parameters': 11180103,
            'checkpoint_rule': 'last epoch',
            'manifest_sha256': '0' * 64,
        }
        self.assertEqual(set(changes), set(PROTOCOL_FIELDS))
        for field, replacement in changes.items():
            with self.subTest(field=field):
                actual = copy.deepcopy(expected)
                actual[field] = replacement
                with self.assertRaisesRegex(ProtocolMismatchError, field):
                    validate_completed_run_config(actual, expected)

    def test_each_missing_protocol_field_is_rejected(self) -> None:
        expected = self.expected()
        for field in PROTOCOL_FIELDS:
            with self.subTest(field=field):
                actual = copy.deepcopy(expected)
                del actual[field]
                with self.assertRaisesRegex(ProtocolMismatchError, field):
                    validate_completed_run_config(actual, expected)

    def test_changed_expected_protocol_is_not_reused(self) -> None:
        actual = self.expected('resnet18_aug')
        expected = copy.deepcopy(actual)
        expected['lr'] = 0.0001
        expected['batch_size'] = 16
        with self.assertRaisesRegex(ProtocolMismatchError, 'batch_size.*lr'):
            validate_completed_run_config(actual, expected, run_name='old_run')

    def test_numeric_type_aliases_are_rejected(self) -> None:
        expected = self.expected()
        for field, alias in [('batch_size', 64.0), ('pretrained', 0), ('seed', 2026.0)]:
            with self.subTest(field=field):
                actual = copy.deepcopy(expected)
                actual[field] = alias
                with self.assertRaisesRegex(ProtocolMismatchError, field):
                    validate_completed_run_config(actual, expected)

    def test_classes_are_copied_without_aliasing(self) -> None:
        classes = list(self.classes)
        expected = build_training_protocol(
            method='cnn', seed=2026, epochs=10, classes=classes,
            parameters=94375, manifest_sha256=self.manifest_sha256,
        )
        classes.reverse()
        self.assertEqual(expected['classes'], self.classes)

    def test_extra_nonprotocol_metadata_is_allowed(self) -> None:
        expected = self.expected()
        actual = dict(expected, note='unrelated metadata')
        validate_completed_run_config(actual, expected)

    def test_unknown_method_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Unknown method'):
            self.expected('resnet18_typo')

    def test_unstructured_config_is_rejected(self) -> None:
        for actual in (None, [], 'not a config'):
            with self.subTest(actual=actual):
                with self.assertRaisesRegex(ProtocolMismatchError, 'mapping'):
                    validate_completed_run_config(actual, self.expected())

    def check_completed_run_branch(
        self, actual: dict[str, object], runtime_parameters: int,
    ) -> tuple[list[object], str]:
        """Exercise the real skip branch with a CPU-model stub, not torch.

        Compiling only ``run`` avoids third-party imports. Any fall-through to
        real training would fail because the stub exposes no training API.
        Test fixture writes are confined to a disposable temporary directory.
        """
        source = ast.parse((ROOT / 'train_experiment.py').read_text(encoding='utf-8'))
        run_node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        module = ast.Module(body=[run_node], type_ignores=[])
        calls = []

        @contextlib.contextmanager
        def fork_rng(*, devices: list[int]):
            calls.append(('fork_rng', devices))
            yield

        def make_model(method: str, pretrained: bool):
            calls.append(('model', method, pretrained))
            return SimpleNamespace(parameters=lambda: [SimpleNamespace(numel=lambda: runtime_parameters)])

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'manifest.json').write_bytes((ROOT / 'manifest.json').read_bytes())
            folder = root / 'runs' / 'cnn_seed2026'
            folder.mkdir(parents=True)
            (folder / 'test.json').write_text('{}')
            (folder / 'config.json').write_text(json.dumps(actual))
            namespace = {
                'ROOT': root, 'CLASSES': self.classes,
                'json': json, 'hashlib': hashlib,
                'torch': SimpleNamespace(random=SimpleNamespace(fork_rng=fork_rng)),
                'make_model': make_model,
                'build_training_protocol': build_training_protocol,
                'validate_completed_run_config': validate_completed_run_config,
            }
            exec(compile(module, 'train_experiment.py', 'exec'), namespace)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                namespace['run']('cnn', 2026, [], 10)
            return calls, stdout.getvalue()

    def test_real_skip_branch_uses_rng_preserving_untrained_model_probe(self) -> None:
        calls, stdout = self.check_completed_run_branch(self.expected(), 94375)
        self.assertEqual(calls, [('fork_rng', []), ('model', 'cnn', False)])
        self.assertIn('already complete cnn_seed2026', stdout)

    def test_real_skip_branch_rejects_changed_runtime_parameter_count(self) -> None:
        with self.assertRaisesRegex(ProtocolMismatchError, 'parameters'):
            self.check_completed_run_branch(self.expected(), 94376)

    def test_real_skip_branch_rejects_stale_saved_learning_rate(self) -> None:
        actual = self.expected()
        actual['lr'] = 0.01
        with self.assertRaisesRegex(ProtocolMismatchError, 'lr'):
            self.check_completed_run_branch(actual, 94375)

    def test_pure_protocol_module_does_not_import_torch(self) -> None:
        source = ast.parse(Path(training_protocol.__file__).read_text())
        imported = set()
        for node in ast.walk(source):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split('.')[0])
        self.assertLessEqual(imported, {'collections'})


if __name__ == '__main__':
    unittest.main()
