"""Independently verify split integrity, checksums and prediction metrics."""
import csv
import hashlib
import itertools
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from verify_artifacts import verify
ROOT=Path(__file__).resolve().parent

def main():
    verify(ROOT)  # Also bind every prediction path/label to the frozen manifest.
    rows=json.loads((ROOT/'manifest.json').read_text())['rows']
    checks={}
    for key in ('path','sha256','pixel_sha256','leaf_group'):
        sets={s:{r[key] for r in rows if r['split']==s} for s in ('train','val','test')}
        overlaps={f'{a}_{b}':len(sets[a]&sets[b]) for a,b in itertools.combinations(sets,2)}
        assert not any(overlaps.values()), (key,overlaps)
        checks[key]=overlaps
    leaf_to_split=defaultdict(set)
    for r in rows:
        for atom in r['leaf_group'].split('|'):
            leaf_to_split[atom].add(r['split'])
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    assert all(len(s)==1 for s in leaf_to_split.values()), 'atomic leaf identity leakage'
    checks['atomic_leaf_ids']=len(leaf_to_split)
    raw=subprocess.check_output(['git','-C',str(ROOT/'source_metadata'),'ls-tree','-rz','--name-only','HEAD','raw/color']).decode().split('\0')
    counts=Counter(p.split('/')[2] for p in raw if '/Tomato' in p and p.lower().endswith(('.jpg','.jpeg','.png')))
    results=[]
    for f in sorted((ROOT/'runs').glob('*/test.json')):
        reported=json.loads(f.read_text()); config=json.loads((f.parent/'config.json').read_text()); classes=config['classes']
        for cond in ('clean','blur','dark'):
            cm=np.zeros((len(classes),len(classes)),int)
            with (f.parent/f'predictions_{cond}.csv').open() as h:
                for row in csv.DictReader(h):
                    cm[classes.index(row['target']),classes.index(row['prediction'])]+=1
            tp=np.diag(cm); p=tp/np.maximum(cm.sum(0),1); r=tp/np.maximum(cm.sum(1),1)
            f1=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=(p+r)>0)
            acc=float(tp.sum()/cm.sum())
            assert cm.sum()==sum(x['split']=='test' for x in rows)
            assert cm.tolist()==reported[cond]['confusion']
            assert abs(acc-reported[cond]['accuracy'])<1e-12
            assert abs(float(f1.mean())-reported[cond]['macro_f1'])<1e-12
            results.append({'method':config['method'],'seed':config['seed'],'condition':cond,'accuracy':acc,'macro_f1':float(f1.mean()),'n':int(cm.sum())})
    report={'split_checks':checks,'raw_tomato_counts':dict(counts),'selected_counts':dict(Counter(r['class_name'] for r in rows)),'validated_runs':len(results)//3,'metric_records':results}
    (ROOT/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='metric_records'},indent=2))
if __name__=='__main__': main()
