"""Reload every final checkpoint and compare fresh clean-test predictions."""
import csv
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import train_experiment as study
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.cuda.set_per_process_memory_fraction(.25)
device=torch.device('cuda:0')
rows=[r for r in json.loads((ROOT/'manifest.json').read_text())['rows'] if r['split']=='test']
verified=[]
for path in sorted((ROOT/'runs').glob('*/best.pt')):
    ck=torch.load(path,map_location='cpu');study.CLASSES=ck['config']['classes']
    model=study.make_model(ck['config']['method'],pretrained=False)
    model.load_state_dict(ck['model']);model.to(device);model.eval()
    with (path.parent/'predictions_clean.csv').open() as h: old=list(csv.DictReader(h))
    loader=DataLoader(study.Leaves(rows),batch_size=64,num_workers=0)
    m,yy,pred,conf=study.evaluate(model,loader,device)
    assert [study.CLASSES[i] for i in pred]==[r['prediction'] for r in old],path
    verified.append({'run':path.parent.name,'n':len(pred),'predictions_identical':True,'accuracy':m['accuracy']})
    print(path.parent.name,'PASS',flush=True)
assert len(verified)==9
(ROOT/'checkpoint_reload_audit.json').write_text(json.dumps(verified,indent=2))
