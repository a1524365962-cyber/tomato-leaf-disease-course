"""Select explicitly illustrative examples from frozen records, not new results."""
import csv,json,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent
rows=json.loads((R/'manifest.json').read_text())['rows']
classes=sorted({x['class_name'] for x in rows})
examples=[sorted([r for r in rows if r['class_name']==c and r['split']=='train'],key=lambda r:r['path'])[0] for c in classes]
def preds(method,condition):
    with (R/'runs'/f'{method}_seed2026'/f'predictions_{condition}.csv').open() as f:return {r['path']:r for r in csv.DictReader(f)}
b={c:preds('resnet18',c) for c in ['clean','dark','blur']};a={c:preds('resnet18_aug',c) for c in b}
cases=[]
for cond,rule in [('dark','rescue'),('blur','failure'),('clean','failure')]:
    candidates=[]
    for path,x in sorted(a[cond].items()):
        y=b[cond][path]
        ok=(x['prediction']==x['target'] and y['prediction']!=y['target']) if rule=='rescue' else (x['prediction']!=x['target'])
        if ok:candidates.append({'path':path,'condition':cond,'rule':rule,'target':x['target'],'baseline':y['prediction'],'augmented':x['prediction']})
    cases+=candidates[:2]
meta={'training_examples':examples,'test_cases':cases,'selection':'lexicographic first per training class; first two matching cases per stated condition, seed 2026; illustrative not prevalence estimate'}
(R/'visual_examples.json').write_text(json.dumps(meta,indent=2))
with tarfile.open(R/'visual_examples.tar.gz','w:gz') as t:
    for p in sorted({r['path'] for r in examples+cases}):t.add(R/p,arcname=p)
    t.add(R/'visual_examples.json',arcname='visual_examples.json')
print(len(examples),len(cases))
