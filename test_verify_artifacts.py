"""Regression checks: intentional record corruption must not pass the audit."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from verify_artifacts import verify, metrics

ROOT = Path(__file__).resolve().parent


class RecordAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('manifest.json', 'data_summary.json', 'results_summary.json', 'metrics_source.csv'):
            shutil.copy2(ROOT / name, self.root / name)
        shutil.copytree(ROOT / 'runs', self.root / 'runs', ignore=shutil.ignore_patterns('*.pt'))

    def test_complete_record_bundle(self) -> None:
        self.assertEqual(verify(self.root)['prediction_files'], 27)

    def test_summary_corruption_is_rejected(self) -> None:
        p = self.root / 'results_summary.json'
        data = json.loads(p.read_text())
        data['methods']['resnet18']['clean']['macro_f1']['mean'] += 0.01
        p.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            verify(self.root)

    def test_missing_run_is_rejected(self) -> None:
        (self.root / 'runs/cnn_seed2026/config.json').unlink()
        with self.assertRaises(FileNotFoundError):
            verify(self.root)

    def test_duplicate_prediction_path_is_rejected(self) -> None:
        p = self.root / 'runs/cnn_seed2026/predictions_clean.csv'
        with p.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        rows[1]['path'] = rows[0]['path']
        with p.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValueError):
            verify(self.root)

    def test_zero_denominator_f1(self) -> None:
        self.assertEqual(metrics([[2, 0], [0, 0]])['f1'], [1.0, 0.0])

    def test_group_count_corruption_is_rejected(self) -> None:
        p=self.root/'data_summary.json'
        data=json.loads(p.read_text());data['n_leaf_groups']+=1;p.write_text(json.dumps(data))
        with self.assertRaises(ValueError): verify(self.root)

    def test_wrong_checkpoint_rule_is_rejected(self) -> None:
        p=self.root/'runs/cnn_seed2026/config.json'
        data=json.loads(p.read_text());data['checkpoint_rule']='last epoch';p.write_text(json.dumps(data))
        with self.assertRaises(ValueError): verify(self.root)


if __name__ == '__main__':
    unittest.main()
