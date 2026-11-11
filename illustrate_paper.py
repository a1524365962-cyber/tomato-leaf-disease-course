"""Original code-aligned diagrams and real-image qualitative analysis."""
import os,sys,json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from PIL import Image
R=Path(__file__).resolve().parent;F=R/'figures';Q=R/'qa'
if os.environ.get('FIGURE_QA_SCRIPTS'):sys.path.insert(0,os.environ['FIGURE_QA_SCRIPTS'])
try:from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:require_matplotlib_panel_alignment=None
mpl.rcParams.update({'font.family':'Arial','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.canvas.draw()
    if require_matplotlib_panel_alignment:require_matplotlib_panel_alignment(fig,json_out=Q/f'{name}.alignment.json',strict=True)
    else:(Q/f'{name}.alignment.json').write_text(json.dumps({'status':'unavailable','reason':'optional external auditor not installed'}))
    fig.savefig(F/f'{name}.pdf');fig.savefig(F/f'{name}.svg');fig.savefig(F/f'{name}.png',dpi=600);plt.close(fig)
def box(ax,x,y,w,h,t,color='#e5eef5'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.007',fc=color,ec='#657b88',lw=.6))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=7)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,lw=.7,color='#647783'))

# One wide diagram: every shape describes the implemented model/protocol.
fig,ax=plt.subplots(figsize=(6.9,3.3));fig.subplots_adjust(left=.02,right=.98,bottom=.03,top=.98);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.01,.965,'(a) Data protocol and training / inference paths',weight='bold',va='top')
xs=[.01,.19,.39,.59,.79];ws=[.14,.16,.16,.16,.19]
labels=['PlantVillage\n7 tomato classes\n3,500 images','Leaf-ID union\n1,799 groups\n2,448 / 512 / 540','RGB 128 x 128\nTrain-only\naugmentation','ImageNet\nResNet18\nFull fine-tuning','Global average pool\n512 -> 7 classifier\nSoftmax prediction']
for i,(x,w,t) in enumerate(zip(xs,ws,labels)):
    box(ax,x,.68,w,.19,t,'#f8e5cc' if i==2 else '#e5eef5')
    if i<4:arrow(ax,(x+w,.775),(xs[i+1],.775))
box(ax,.60,.48,.16,.10,'Cross-entropy\nAdamW + cosine','#f6eccb');ax.plot([.89,.89,.76],[.68,.53,.53],color='#647783',lw=.7);arrow(ax,(.80,.53),(.76,.53))
ax.text(.01,.57,'Inference: deterministic resize and normalization; no random augmentation.',fontsize=7)
ax.text(.01,.43,'(b) ResNet18 feature sizes and a basic residual block',weight='bold')
stage=['Stem\n64 x 32 x 32','Layer 1\n64 x 32 x 32','Layer 2\n128 x 16 x 16','Layer 3\n256 x 8 x 8','Layer 4\n512 x 4 x 4']
for i,t in enumerate(stage):
    x=.01+i*.196;box(ax,x,.25,.16,.12,t)
    if i<4:arrow(ax,(x+.16,.31),(x+.196,.31))
ax.text(.01,.15,'Basic block',weight='bold');box(ax,.19,.07,.21,.10,'Conv 3x3 - BN - ReLU','#e7efdc');box(ax,.49,.07,.18,.10,'Conv 3x3 - BN','#e7efdc');box(ax,.77,.07,.19,.10,'Add shortcut - ReLU','#f8e5cc')
arrow(ax,(.15,.12),(.19,.12));arrow(ax,(.40,.12),(.49,.12));arrow(ax,(.67,.12),(.77,.12))
ax.plot([.16,.16,.73,.73],[.12,.215,.215,.12],color='#647783',lw=.7);arrow(ax,(.73,.12),(.77,.12))
save(fig,'model_overview')

meta=json.loads((R/'visual_examples.json').read_text());classes=sorted({r['class_name'] for r in meta['training_examples']});short=dict(zip(classes,['BS','EB','LB','LM','SL','YL','H']))
fig,axes=plt.subplots(2,4,figsize=(3.35,2.15));fig.subplots_adjust(left=.01,right=.99,bottom=.03,top=.88,wspace=.12,hspace=.34)
for i,ax in enumerate(axes.flat):
    ax.set_axis_off()
    if i<7:
        r=meta['training_examples'][i];ax.imshow(Image.open(R/r['path']).convert('RGB'),aspect='auto');ax.set_title(short[r['class_name']],fontsize=7,pad=3)
    else:ax.text(.5,.5,'PlantVillage\ntraining samples',ha='center',va='center',fontsize=6)
save(fig,'leaf_examples')

cam=np.load(R/'cams.npz');cm=json.loads((R/'cams_metadata.json').read_text());indices=[0,2]
fig,axes=plt.subplots(2,3,figsize=(6.9,4.5));fig.subplots_adjust(left=.02,right=.98,bottom=.03,top=.93,wspace=.08,hspace=.22)
for row,j in enumerate(indices):
    case=cm['cases'][j];rgb=cam[f'input_{j}']/255
    for col,ax in enumerate(axes[row]):
        ax.axis('off');ax.imshow(rgb,aspect='auto')
        if col:
            key=['baseline','augmented'][col-1];ax.imshow(cam[f'{key}_{j}'],cmap='magma',vmin=0,vmax=1,alpha=.45,aspect='auto')
            title=('ResNet18' if col==1 else 'ResNet18 + augmentation')+' | Pred: '+short[case[key]]
        else:title=f"({chr(97+row)}) {case['condition'].capitalize()} input | True: {short[case['target']]}"
        ax.set_title(title,fontsize=7,pad=4)
save(fig,'gradcam_cases')
print('Rendered method, dataset and paired Grad-CAM figures')
