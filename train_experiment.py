"""Reproducible seven-class tomato leaf experiment. No synthetic metrics."""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from training_protocol import (
    BATCH_SIZE, INPUT_SIZE, WEIGHT_DECAY, build_training_protocol,
    learning_rate_for_method, validate_completed_run_config,
)

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parent
os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = []
def metrics(target, pred, n):
    cm = np.zeros((n,n),dtype=np.int64)
    for a,b in zip(target,pred):
        cm[a,b] += 1
    tp = np.diag(cm)
    precision = tp / np.maximum(cm.sum(0),1)
    recall = tp / np.maximum(cm.sum(1),1)
    f1 = 2*precision*recall / np.maximum(precision+recall,1e-12)
    return {'accuracy':float(tp.sum()/max(cm.sum(),1)), 'macro_f1':float(f1.mean()), 'precision':precision.tolist(),'recall':recall.tolist(),'f1':f1.tolist(),'confusion':cm.tolist(),'n':int(cm.sum())}

class Leaves(Dataset):
    def __init__(self, rows, augment=False, condition='clean'):
        self.rows = rows
        self.images = []
        self.condition = condition
        self.tf = transforms.Compose(([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(20), transforms.ColorJitter(brightness=.2,contrast=.2,saturation=.1)] if augment else []) + [transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        for r in rows:
            with Image.open(ROOT/r['path']) as im:
                self.images.append(im.convert('RGB').resize((INPUT_SIZE,INPUT_SIZE),Image.Resampling.BILINEAR))
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,i):
        im = self.images[i].copy()
        if self.condition == 'blur':
            im = im.filter(ImageFilter.GaussianBlur(1.5))
        elif self.condition == 'dark':
            im = ImageEnhance.Brightness(im).enhance(.6)
        return self.tf(im), CLASSES.index(self.rows[i]['class_name']), i

class SmallCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        layers=[]
        prev=3
        for c in (32,64,128):
            layers += [nn.Conv2d(prev,c,3,padding=1,bias=False),nn.BatchNorm2d(c),nn.ReLU(),nn.MaxPool2d(2)]
            prev=c
        self.features = nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(128,n)
    def forward(self,x):
        return self.fc(self.features(x).flatten(1))

def make_model(method, pretrained=True):
    if method=='cnn':
        return SmallCNN(len(CLASSES))
    m=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc=nn.Linear(m.fc.in_features,len(CLASSES))
    return m

def evaluate(model,loader,device):
    model.eval()
    yy,pp,confidence=[],[],[]
    loss_sum=0.
    with torch.no_grad():
        for x,y,_ in loader:
            logits=model(x.to(device))
            loss_sum+=nn.functional.cross_entropy(logits,y.to(device),reduction='sum').item()
            prob=logits.softmax(1)
            yy.extend(y.tolist()); pp.extend(logits.argmax(1).cpu().tolist()); confidence.extend(prob.max(1).values.cpu().tolist())
    m=metrics(yy,pp,len(CLASSES)); m['loss']=loss_sum/len(yy)
    return m,yy,pp,confidence

def run(method,seed,rows,epochs):
    out=ROOT/'runs'/f'{method}_seed{seed}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'test.json').exists():
        old=json.loads((out/'config.json').read_text())
        # Count the current CPU architecture without downloading pretrained
        # weights or advancing the RNG used by any subsequent real training.
        with torch.random.fork_rng(devices=[]):
            probe=make_model(method,pretrained=False)
            parameters=sum(p.numel() for p in probe.parameters())
        del probe
        expected=build_training_protocol(
            method=method,seed=seed,epochs=epochs,classes=CLASSES,
            parameters=parameters,
            manifest_sha256=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),
        )
        validate_completed_run_config(old,expected,run_name=out.name)
        print('already complete',out.name,flush=True); return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    trainrows=[r for r in rows if r['split']=='train']
    valrows=[r for r in rows if r['split']=='val']
    testrows=[r for r in rows if r['split']=='test']
    generator=torch.Generator().manual_seed(seed)
    train=DataLoader(Leaves(trainrows,method=='resnet18_aug'),batch_size=BATCH_SIZE,shuffle=True,num_workers=2,pin_memory=True,generator=generator)
    val=DataLoader(Leaves(valrows),batch_size=BATCH_SIZE,num_workers=2,pin_memory=True)
    device=torch.device('cuda:0')
    model=make_model(method).to(device)
    lr=learning_rate_for_method(method)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    config=build_training_protocol(
        method=method,seed=seed,epochs=epochs,classes=CLASSES,
        parameters=sum(p.numel() for p in model.parameters()),
        manifest_sha256=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),
    )
    (out/'config.json').write_text(json.dumps(config,indent=2))
    best=-1.; start=time.perf_counter(); history=[]
    for epoch in range(1,epochs+1):
        model.train(); total=0; correct=0; loss_sum=0.; epoch_start=time.perf_counter()
        for x,y,_ in train:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits=model(x); loss=nn.functional.cross_entropy(logits,y)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            loss.backward(); opt.step()
            total+=len(y); correct+=(logits.argmax(1)==y).sum().item(); loss_sum+=loss.item()*len(y)
        vm,*_=evaluate(model,val,device)
        rec={'epoch':epoch,'train_loss':loss_sum/total,'train_accuracy':correct/total,'val_loss':vm['loss'],'val_accuracy':vm['accuracy'],'val_macro_f1':vm['macro_f1'],'lr':opt.param_groups[0]['lr'],'seconds':time.perf_counter()-epoch_start}
        history.append(rec)
        with (out/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rec));w.writeheader();w.writerows(history)
        if vm['macro_f1']>best:
            best=vm['macro_f1']
            torch.save({'model':model.state_dict(),'config':config,'epoch':epoch,'val_metrics':vm},out/'best.pt')
        sched.step()
        print(out.name,json.dumps(rec),flush=True)
    train_seconds=time.perf_counter()-start
    ck=torch.load(out/'best.pt',map_location=device)
    model.load_state_dict(ck['model'])
    results={'best_epoch':ck['epoch'],'train_seconds':train_seconds}
    for condition in ('clean','blur','dark'):
        loader=DataLoader(Leaves(testrows,condition=condition),batch_size=BATCH_SIZE,num_workers=2,pin_memory=True)
        m,yy,pp,conf=evaluate(model,loader,device)
        results[condition]=m
        with (out/f'predictions_{condition}.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['path','target','prediction','softmax_max'])
            w.writerows((r['path'],CLASSES[y],CLASSES[p],c) for r,y,p,c in zip(testrows,yy,pp,conf))
    (out/'test.json').write_text(json.dumps(results,indent=2))
    print('COMPLETE',out.name,json.dumps(results),flush=True)
    del model,opt,train,val
    torch.cuda.empty_cache()

def main():
    global CLASSES
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=10);p.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028]);p.add_argument('--methods',nargs='+',default=['cnn','resnet18','resnet18_aug']);args=p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.25)
    m=json.loads((ROOT/'manifest.json').read_text())
    CLASSES=sorted({r['class_name'] for r in m['rows']})
    env={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'physical_gpu':'CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'],'n_cpu_threads':4,'cuda_memory_fraction':.25}
    (ROOT/'environment.json').write_text(json.dumps(env,indent=2))
    for seed in args.seeds:
        for method in args.methods:
            run(method,seed,m['rows'],args.epochs)
    print('ALL RUNS COMPLETE',flush=True)
if __name__=='__main__':
    main()
