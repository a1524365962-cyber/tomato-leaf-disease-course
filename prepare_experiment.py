"""Fetch a bounded, provenance-tracked PlantVillage tomato subset, not a full archive."""
import concurrent.futures as cf
import hashlib
import io
import json
import random
import subprocess
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
META = ROOT / 'source_metadata'
def git(*args):
    return subprocess.check_output(['git', '-C', str(META), *args])

def restore_frozen_manifest(manifest_path):
    """Restore exact published files without rewriting a frozen split."""
    frozen=json.loads(manifest_path.read_text(encoding='utf-8'))
    commit=frozen['commit']
    if commit != '7f7ecc7e1eaca78107e3affe7cb5abd9427e139a':
        raise ValueError('frozen manifest has an unexpected source commit')
    def restore(row):
        dest=(ROOT/row['path']).resolve()
        if not dest.is_relative_to((ROOT/'data').resolve()):
            raise ValueError('frozen data path is outside data/')
        if dest.exists():
            content=dest.read_bytes()
        else:
            url='https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/'+commit+'/'+urllib.parse.quote(row['source_path'])
            for attempt in range(5):
                try:
                    content=urllib.request.urlopen(url,timeout=40).read()
                    break
                except (OSError,TimeoutError):
                    if attempt==4: raise
                    time.sleep(1+attempt)
        if hashlib.sha256(content).hexdigest()!=row['sha256']:
            raise ValueError('original image SHA256 mismatch: '+row['path'])
        with Image.open(io.BytesIO(content)) as source:
            im=source.convert('RGB');im.load()
        if hashlib.sha256(im.tobytes()).hexdigest()!=row['pixel_sha256']:
            raise ValueError('decoded pixel SHA256 mismatch; use the recorded Pillow 9.4.0 decoder: '+row['path'])
        if (im.width,im.height)!=(row['width'],row['height']):
            raise ValueError('image dimensions mismatch: '+row['path'])
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(content)
    with cf.ThreadPoolExecutor(max_workers=12) as pool:
        for i,_ in enumerate(pool.map(restore,frozen['rows'])):
            if i%200==0: print('restored/verified',i+1,'/',len(frozen['rows']),flush=True)
    print('Frozen manifest preserved; no selection or split was regenerated.',flush=True)

def main():
    manifest_path=ROOT/'manifest.json'
    if manifest_path.exists():
        restore_frozen_manifest(manifest_path)
        return
    commit = git('rev-parse', 'HEAD').decode().strip()
    assert commit == '7f7ecc7e1eaca78107e3affe7cb5abd9427e139a', 'metadata commit differs from the frozen protocol'
    mapping = json.loads(git('show', 'HEAD:leaf-map.json'))
    paths = git('ls-tree', '-rz', '--name-only', 'HEAD', 'raw/color').decode().split('\0')
    cats = defaultdict(list)
    for p in paths:
        if '/Tomato' not in p or not p.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        cat = p.split('/')[2]
        if 'Spider_mites' in cat:
            continue  # insect injury is outside this disease-only course study
        stem = Path(p).stem.split('___', 1)[-1].lower()
        labels = mapping.get(stem)
        if not labels:
            continue  # do not infer unknown leaf identities
        group = '|'.join(sorted(labels))
        cats[cat].append({'source_path': p, 'class_name': cat, 'leaf_group': group})
    assert len(cats) == 7, sorted(cats)
    chosen = []
    for cat, rows in sorted(cats.items()):
        rng = random.Random(2026)
        rng.shuffle(rows)
        chosen.extend(rows[:500])
    out = ROOT / 'data'
    out.mkdir(exist_ok=True)
    def fetch(row):
        dest = out / row['class_name'] / Path(row['source_path']).name
        dest.parent.mkdir(exist_ok=True)
        url = 'https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/' + commit + '/' + urllib.parse.quote(row['source_path'])
        for attempt in range(5):
            try:
                content = dest.read_bytes() if dest.exists() else urllib.request.urlopen(url, timeout=40).read()
                im = Image.open(io.BytesIO(content)).convert('RGB')
                im.load()
                if not dest.exists():
                    dest.write_bytes(content)
                row.update(path=str(dest.relative_to(ROOT)), sha256=hashlib.sha256(content).hexdigest(), pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest(), width=im.width, height=im.height)
                return row
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        got = []
        for i, row in enumerate(ex.map(fetch, chosen)):
            got.append(row)
            if i % 200 == 0:
                print('downloaded', i+1, '/', len(chosen), flush=True)
    # Union exact pixel duplicates and dataset-provided leaf IDs before splitting.
    parent = list(range(len(got)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    lookup = {}
    for i, row in enumerate(got):
        # A source image can list multiple candidate leaf identities. Union
        # each identity, not just the joined string, to close partial overlap.
        for key in [('leaf', atom) for atom in row['leaf_group'].split('|')] + [('pixels', row['pixel_sha256'])]:
            if key in lookup:
                parent[root(i)] = root(lookup[key])
            else:
                lookup[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(got):
        groups[root(i)].append(row)
    byclass = defaultdict(list)
    for group in groups.values():
        assert len({r['class_name'] for r in group}) == 1, 'cross-class duplicate'
        byclass[group[0]['class_name']].append(group)
    rows = []
    for cat, gs in sorted(byclass.items()):
        random.Random(2026).shuffle(gs)
        n = len(gs)
        for j, group in enumerate(gs):
            split = 'train' if j < int(n*.7) else ('val' if j < int(n*.85) else 'test')
            for r in group:
                r['split'] = split
                rows.append(r)
    assert not ({r['leaf_group'] for r in rows if r['split']=='train'} & {r['leaf_group'] for r in rows if r['split']=='test'})
    (ROOT/'manifest.json').write_text(json.dumps({'commit':commit,'selection_seed':2026,'cap_per_class':500,'rows':rows},indent=2),encoding='utf-8')
    summary = {'n_images':len(rows),'n_leaf_groups':len(groups),'split_counts':dict(Counter(r['split'] for r in rows)), 'by_class':{c:dict(Counter(r['split'] for r in rows if r['class_name']==c)) for c in sorted(cats)},'mapped_candidates':{k:len(v) for k,v in cats.items()},'exact_pixel_duplicate_count':len(rows)-len({r['pixel_sha256'] for r in rows})}
    (ROOT/'data_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':
    main()
