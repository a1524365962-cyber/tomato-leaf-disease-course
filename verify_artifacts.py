"""Verify the published records offline using only Python's standard library.

This does not re-run a model or validate unavailable original images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import tarfile
from training_protocol import build_training_protocol, validate_completed_run_config

METHODS = ('cnn', 'resnet18', 'resnet18_aug')
SEEDS = (2026, 2027, 2028)
CONDITIONS = ('clean', 'blur', 'dark')
WEIGHT_SHA256 = '92285ee5a12593adb4eb6b98fa38fec6dead5164411951bf3ada73891f3fbe46'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def near(a: float, b: float, label: str) -> None:
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b) < 1e-10, label)


def aggregate(actual: list[float], saved: dict, label: str) -> None:
    require(len(actual) == len(saved['values']), label + ': number of values')
    for a, b in zip(actual, saved['values']):
        near(a, b, label + ': seed value')
    near(mean(actual), saved['mean'], label + ': mean')
    near(stdev(actual), saved['sd'], label + ': sample SD')


def metrics(matrix: list[list[int]]) -> dict:
    n = len(matrix)
    total = sum(map(sum, matrix))
    precision = [matrix[k][k] / max(sum(row[k] for row in matrix), 1) for k in range(n)]
    recall = [matrix[k][k] / max(sum(matrix[k]), 1) for k in range(n)]
    f1 = [2*p*r/(p+r) if p+r else 0.0 for p, r in zip(precision, recall)]
    return {'accuracy': sum(matrix[k][k] for k in range(n))/total,
            'macro_f1': mean(f1), 'precision': precision, 'recall': recall,
            'f1': f1, 'confusion': matrix, 'n': total}


def verify(root: Path) -> dict:
    manifest_path = root / 'manifest.json'
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    rows = load(manifest_path)['rows']
    data = load(root / 'data_summary.json')
    summary = load(root / 'results_summary.json')
    classes = sorted({r['class_name'] for r in rows})
    require(len(classes) == 7 and len(rows) == 3500, 'dataset size/classes')
    require(data['n_images'] == len(rows), 'summary dataset size')
    require(len({r['path'] for r in rows}) == len(rows), 'duplicate manifest paths')
    require(summary['manifest_sha256'] == digest, 'summary manifest SHA256')
    require(summary['n_runs'] == 9 and summary['seeds'] == list(SEEDS), 'summary run protocol')
    atomic_splits: dict[str, set[str]] = {}
    for key in ('path', 'sha256', 'pixel_sha256', 'leaf_group'):
        sets = [{r[key] for r in rows if r['split'] == split} for split in ('train', 'val', 'test')]
        require(all(not sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)), key + ': split leakage')
    for row in rows:
        require(row['split'] in ('train', 'val', 'test'), 'unknown split')
        for atom in row['leaf_group'].split('|'):
            atomic_splits.setdefault(atom, set()).add(row['split'])
    require(all(len(v) == 1 for v in atomic_splits.values()), 'atomic leaf identity leakage')
    parent=list(range(len(rows)))
    def find(i: int) -> int:
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    lookup={}
    for i,row in enumerate(rows):
        for key in [('leaf',v) for v in row['leaf_group'].split('|')]+[('pixel',row['pixel_sha256'])]:
            if key in lookup: parent[find(i)]=find(lookup[key])
            else: lookup[key]=i
    require(len({find(i) for i in range(len(rows))}) == data['n_leaf_groups'] == 1799, 'connected leaf group count')
    for split, expected in [('train', 2448), ('val', 512), ('test', 540)]:
        count = sum(r['split'] == split for r in rows)
        require(count == expected == data['split_counts'][split], split + ': size mismatch')
        for c in classes:
            require(sum(r['split'] == split and r['class_name'] == c for r in rows) == data['by_class'][c][split], c + ': class count')
    test_rows = {r['path']: r['class_name'] for r in rows if r['split'] == 'test'}
    results: dict[tuple[str, int, str], dict] = {}
    configs = {}
    tests = {}
    for method in METHODS:
        for seed in SEEDS:
            folder = root / 'runs' / f'{method}_seed{seed}'
            cfg = load(folder / 'config.json')
            expected=build_training_protocol(method=method,seed=seed,epochs=10,classes=classes,
                parameters=94375 if method=='cnn' else 11180103,manifest_sha256=digest)
            validate_completed_run_config(cfg,expected,run_name=folder.name)
            test = load(folder / 'test.json')
            configs[method, seed] = cfg
            tests[method, seed] = test
            require(cfg['method'] == method and cfg['seed'] == seed, 'run identity')
            require(cfg['manifest_sha256'] == digest and cfg['classes'] == classes, 'run manifest/classes')
            require(cfg['epochs'] == 10 and cfg['batch_size'] == 64 and cfg['input_size'] == 128, 'training budget')
            require(cfg['optimizer'] == 'AdamW' and cfg['scheduler'] == 'cosine', 'optimizer/scheduler')
            near(cfg['weight_decay'], 1e-4, 'weight decay')
            near(cfg['lr'], 1e-3 if method == 'cnn' else 3e-4, 'initial learning rate')
            require(cfg['parameters'] == (94375 if method == 'cnn' else 11180103), 'parameter count')
            require(cfg['pretrained'] == (method != 'cnn'), 'initialization')
            with (folder / 'history.csv').open(newline='', encoding='utf-8') as handle:
                history = list(csv.DictReader(handle))
            require([int(h['epoch']) for h in history] == list(range(1, 11)), 'epoch history')
            require(all(math.isfinite(float(v)) for h in history for v in h.values()), 'nonfinite history')
            best = max(history, key=lambda h: float(h['val_macro_f1']))
            require(int(best['epoch']) == test['best_epoch'], 'validation checkpoint selection')
            for condition in CONDITIONS:
                with (folder / f'predictions_{condition}.csv').open(newline='', encoding='utf-8') as handle:
                    predictions = list(csv.DictReader(handle))
                require(len(predictions) == len(test_rows), 'prediction record count')
                require(len({p['path'] for p in predictions}) == len(test_rows), 'duplicate prediction paths')
                require({p['path'] for p in predictions} == set(test_rows), 'prediction paths do not match held-out set')
                matrix = [[0]*len(classes) for _ in classes]
                for p in predictions:
                    require(p['target'] == test_rows[p['path']], 'prediction target differs from manifest')
                    require(p['prediction'] in classes, 'unknown predicted class')
                    require(0 <= float(p['softmax_max']) <= 1, 'softmax range')
                    matrix[classes.index(p['target'])][classes.index(p['prediction'])] += 1
                actual = metrics(matrix)
                saved = test[condition]
                require(actual['confusion'] == saved['confusion'] and actual['n'] == saved['n'], 'confusion/count mismatch')
                for field in ('accuracy', 'macro_f1'):
                    near(actual[field], saved[field], field + ': prediction metric')
                for field in ('precision', 'recall', 'f1'):
                    require(len(actual[field]) == len(saved[field]), field + ': class count')
                    for a, b in zip(actual[field], saved[field]):
                        near(a, b, field + ': class metric')
                results[method, seed, condition] = actual
    for method in METHODS:
        saved = summary['methods'][method]
        for condition in CONDITIONS:
            for field in ('accuracy', 'macro_f1'):
                aggregate([results[method, s, condition][field] for s in SEEDS], saved[condition][field], method + '/' + condition + '/' + field)
        require(saved['parameters'] == configs[method, SEEDS[0]]['parameters'], 'summary parameters')
        require(saved['best_epochs'] == [tests[method, s]['best_epoch'] for s in SEEDS], 'summary best epochs')
        aggregate([tests[method, s]['train_seconds'] for s in SEEDS], saved['train_seconds'], 'training time')
        for i, c in enumerate(classes):
            aggregate([results[method, s, 'clean']['f1'][i] for s in SEEDS], summary['per_class'][method][c], c + '/F1')
    for condition in CONDITIONS:
        aggregate([100*(results['resnet18_aug', s, condition]['macro_f1']-results['resnet18', s, condition]['macro_f1']) for s in SEEDS], summary['paired_delta'][condition], condition + '/paired delta')
    with (root / 'metrics_source.csv').open(newline='', encoding='utf-8') as handle:
        sources = list(csv.DictReader(handle))
    keys = [(r['method'], int(r['seed']), r['condition']) for r in sources]
    require(len(keys) == len(set(keys)) == 27 and set(keys) == set(results), 'source metric coverage')
    for row, key in zip(sources, keys):
        require(int(row['n']) == results[key]['n'], 'source metric n')
        for field in ('accuracy', 'macro_f1'):
            near(float(row[field]), results[key][field], 'source metric ' + field)
    return {'runs': 9, 'prediction_files': 27, 'predictions_per_file': 540,
            'manifest_sha256': digest, 'verdict': 'PASS',
            'scope': 'records and summaries only; no new inference or raw-image validation'}


def verify_archive(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    require(digest.hexdigest() == WEIGHT_SHA256, 'checkpoint archive SHA256')
    expected = {f'runs/{m}_seed{s}/best.pt' for m in METHODS for s in SEEDS}
    with tarfile.open(path, 'r:gz') as archive:
        actual = [m.name for m in archive.getmembers() if m.isfile()]
    require(len(actual) == 9 and set(actual) == expected, 'checkpoint archive members')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--checkpoint-archive', type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    if args.checkpoint_archive:
        verify_archive(args.checkpoint_archive)
        result['checkpoint_archive'] = 'SHA256 and nine member paths PASS'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
