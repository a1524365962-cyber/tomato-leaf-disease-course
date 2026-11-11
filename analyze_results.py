"""Aggregate real prediction metrics and draw reproducible paper figures."""
import csv
import hashlib
import json
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
QA=ROOT/'qa'; QA.mkdir(exist_ok=True)
AUDIT=os.environ.get('FIGURE_QA_SCRIPTS')
if AUDIT: sys.path.insert(0,AUDIT)
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment=None
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.3,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
METHODS=['cnn','resnet18','resnet18_aug']; SEEDS=[2026,2027,2028]
LABELS=['Small CNN','ResNet18','ResNet18 + augmentation']
COLORS=['#9199a4','#4377a5','#c66a35']
records={}
manifest_hash=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
for method in METHODS:
    for seed in SEEDS:
        folder=ROOT/'runs'/f'{method}_seed{seed}'
        cfg=json.loads((folder/'config.json').read_text())
        assert cfg['manifest_sha256']==manifest_hash
        assert cfg['epochs']==10
        records[method,seed]=(json.loads((folder/'test.json').read_text()),cfg)

def avg_sd(values):
    a=np.asarray(values,float)
    return {'mean':float(a.mean()),'sd':float(a.std(ddof=1)),'values':a.tolist()}

summary={'seeds':SEEDS,'n_runs':len(records),'manifest_sha256':manifest_hash,'methods':{},'paired_delta':{},'per_class':{}}
source=[]
for method in METHODS:
    r=summary['methods'][method]={}
    for condition in ['clean','blur','dark']:
        r[condition]={metric:avg_sd([records[method,s][0][condition][metric] for s in SEEDS]) for metric in ['accuracy','macro_f1']}
        for seed in SEEDS:
            t=records[method,seed][0][condition]
            source.append({'method':method,'seed':seed,'condition':condition,'accuracy':t['accuracy'],'macro_f1':t['macro_f1'],'n':t['n']})
    r['train_seconds']=avg_sd([records[method,s][0]['train_seconds'] for s in SEEDS])
    r['parameters']=records[method,SEEDS[0]][1]['parameters']
    r['best_epochs']=[records[method,s][0]['best_epoch'] for s in SEEDS]
classes=records[METHODS[0],SEEDS[0]][1]['classes']
for method in METHODS:
    summary['per_class'][method]={c:avg_sd([records[method,s][0]['clean']['f1'][i] for s in SEEDS]) for i,c in enumerate(classes)}
for cond in ['clean','blur','dark']:
    summary['paired_delta'][cond]=avg_sd([100*(records['resnet18_aug',s][0][cond]['macro_f1']-records['resnet18',s][0][cond]['macro_f1']) for s in SEEDS])
(ROOT/'results_summary.json').write_text(json.dumps(summary,indent=2))
with (ROOT/'metrics_source.csv').open('w',newline='') as h:
    w=csv.DictWriter(h,fieldnames=list(source[0]));w.writeheader();w.writerows(source)

def save(fig,name):
    fig.canvas.draw()
    if require_matplotlib_panel_alignment:
        require_matplotlib_panel_alignment(fig,json_out=QA/f'{name}.alignment.json',tolerance_pt=1.5,strict=True)
    else:
        (QA/f'{name}.alignment.json').write_text(json.dumps({'status':'unavailable','reason':'optional external layout auditor not installed'}))
    fig.savefig(FIG/f'{name}.pdf')
    fig.savefig(FIG/f'{name}.svg')
    fig.savefig(FIG/f'{name}.png',dpi=600)
    plt.close(fig)

# Claim: optimization trajectories differ under the same frozen data split.
# Mean +/- sample SD across three optimization seeds; not sampling uncertainty.
fig,ax=plt.subplots(figsize=(3.35,2.4))
fig.subplots_adjust(left=.15,right=.98,bottom=.18,top=.73)
for method,label,color in zip(METHODS,LABELS,COLORS):
    histories=[]
    for seed in SEEDS:
        with (ROOT/'runs'/f'{method}_seed{seed}'/'history.csv').open() as h: histories.append([float(r['val_macro_f1'])*100 for r in csv.DictReader(h)])
    a=np.array(histories);mean=a.mean(0);sd=a.std(0,ddof=1);epochs=np.arange(1,11)
    ax.plot(epochs,mean,label=label,color=color,lw=1.3)
    ax.fill_between(epochs,mean-sd,mean+sd,color=color,alpha=.13,lw=0)
ax.set(xlabel='Epoch',ylabel='Validation macro-F1 (%)',xlim=(1,10),ylim=(40,102),xticks=[1,3,5,7,10])
fig.legend(loc='upper center',bbox_to_anchor=(.54,.99),ncol=1,frameon=False)
save(fig,'validation_curve')

# Claim: class-specific errors remain despite high clean-set aggregate scores.
# Counts are from the prespecified seed 2026, not the best test seed.
cm=np.array(records['resnet18_aug',2026][0]['clean']['confusion'])
fig,ax=plt.subplots(figsize=(3.35,3.15))
fig.subplots_adjust(left=.16,right=.98,bottom=.17,top=.98)
ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(cm.max(),1))
abbr=['BS','EB','LB','LM','SL','YL','H']
ax.set(xticks=range(7),yticks=range(7),xticklabels=abbr,yticklabels=abbr,xlabel='Predicted class',ylabel='True class')
ax.tick_params(length=0)
for i in range(7):
    for j in range(7): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7,color='white' if cm[i,j]>cm.max()/2 else '#222222')
save(fig,'confusion_seed2026')

# Claim: augmentation's effect depends on the test condition.
# Every point is a matched-seed difference on exactly the same 540 test images.
fig,ax=plt.subplots(figsize=(3.35,2.5));fig.subplots_adjust(left=.18,right=.98,bottom=.19,top=.95)
for i,cond in enumerate(['clean','blur','dark']):
    vals=summary['paired_delta'][cond]['values']
    ax.scatter(i+np.array([-.075,0,.075]),vals,c=COLORS[2],s=18,zorder=3)
    ax.errorbar(i+.16,np.mean(vals),yerr=np.std(vals,ddof=1),fmt='s',color='#222222',ms=3,capsize=2,lw=.8,zorder=4)
ax.axhline(0,color='#777777',lw=.7,ls='--',zorder=0)
ax.set(xticks=[0,1,2],xticklabels=['Clean','Blur','Dark'],xlim=(-.4,2.5),ylabel='Augmentation effect on macro-F1 (pp)')
save(fig,'paired_augmentation')
print(json.dumps(summary['methods'],indent=2))
