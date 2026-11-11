"""Code-aligned vector method figure with real inputs and schematic tensors."""
import json
import os
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle
from PIL import Image, ImageEnhance, ImageOps

ROOT=Path(__file__).resolve().parent
if os.environ.get('FIGURE_QA_SCRIPTS'):
    sys.path.insert(0,os.environ['FIGURE_QA_SCRIPTS'])
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment=None

INK='#233D4D'; EDGE='#9CB2BE'; BLUE='#4B7E9D'; TEAL='#4F9489'; ORANGE='#BC7742'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def label(ax,x,y,text,size=7,color=INK,bold=False,ha='center',va='center'):
    return ax.text(x,y,text,ha=ha,va=va,fontsize=size,color=color,weight='bold' if bold else 'normal',zorder=10)

def card(ax,x,y,w,h,text,fill='#EDF3F6',color=INK,size=7):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.15,rounding_size=0.8',
        facecolor=fill,edgecolor=EDGE,linewidth=.65,zorder=2))
    label(ax,x+w/2,y+h/2,text,size=size,color=color)

def arrow(ax,points,color=EDGE,dashed=False,head=5):
    if len(points)>2:
        ax.plot([p[0] for p in points[:-1]],[p[1] for p in points[:-1]],
            color=color,lw=.9,ls=(0,(2,2)) if dashed else '-',zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=head,
        color=color,lw=.9,linestyle=(0,(2,2)) if dashed else '-',zorder=3))

def tensor(ax,cx,cy,w,h,depth=4,color='#BDD3DF'):
    # Stacked planes are topology illustrations, not measured activation maps.
    for i in range(depth-1,-1,-1):
        dx=i*.55;dy=i*.45
        ax.add_patch(Polygon([(cx-w/2+dx,cy-h/2+dy),(cx+w/2+dx,cy-h/2+dy),
            (cx+w/2+dx+1.0,cy+h/2+dy),(cx-w/2+dx+1.0,cy+h/2+dy)],
            closed=True,fc=color,ec=BLUE,lw=.55,zorder=3+depth-i))

def photo(ax,image,x,y,w,h):
    ax.add_patch(Rectangle((x-.15,y-.2),w+.3,h+.4,fc='white',ec=EDGE,lw=.55,zorder=4))
    ax.imshow(image,extent=(x,x+w,y,y+h),aspect='auto',zorder=5)

def main():
    fig=plt.figure(figsize=(6.9,4.05))
    gs=fig.add_gridspec(2,3,height_ratios=[1.22,1],left=.02,right=.98,top=.98,bottom=.035,hspace=.12,wspace=.16)
    a=fig.add_subplot(gs[0,:]);b=fig.add_subplot(gs[1,:2]);c=fig.add_subplot(gs[1,2])
    for ax in [a,b,c]:ax.set(xlim=(0,100));ax.axis('off')
    a.set_ylim(0,47);b.set_ylim(0,54);c.set_ylim(0,100)
    label(a,0,46,'a  Leakage-aware training and evaluation',size=8.5,bold=True,ha='left',va='top')
    meta=json.loads((ROOT/'visual_examples.json').read_text())
    ims=[Image.open(ROOT/r['path']).convert('RGB') for r in meta['training_examples'][:3]]
    # Real source images; resize/augmentation thumbnails are illustrative only.
    photo(a,ims[2],1,23,7.7,11.5);photo(a,ims[1],2.7,21.2,7.7,11.5);photo(a,ims[0],4.4,19.4,7.7,11.5)
    label(a,6.6,15.8,'PlantVillage',bold=True)
    label(a,6.6,11.9,'3,500 images\n7 classes',size=6.8)
    label(a,6.6,6.8,'RGB 128 x 128',size=6.4,color='#657B88')
    arrow(a,[(12.7,27),(15.4,27)])
    groups=[[(17,31),(19,33),(20,29)],[(22,25),(24,27),(24,23)],[(17,22),(19,20)]]
    for pts in groups:
        for p,q in zip(pts,pts[1:]):a.plot([p[0],q[0]],[p[1],q[1]],lw=.7,color=TEAL,zorder=1)
        for x,y in pts:a.add_patch(Circle((x,y),.85,fc='#DCECE6',ec=TEAL,lw=.65,zorder=2))
    label(a,20.5,16.4,'Leaf-ID union',bold=True,size=6.8)
    label(a,20.5,12.5,'1,799 groups',size=6.5)
    arrow(a,[(25,27),(26.5,27),(26.5,35),(28,35)])
    arrow(a,[(26.5,27),(26.5,25),(28,25)])
    arrow(a,[(26.5,25),(26.5,15),(28,15)])
    card(a,28,31.3,11,7.4,'Train  2,448',fill='#F9ECDD',color=ORANGE,size=6.8)
    card(a,28,21.3,11,7.4,'Val  512',size=6.8)
    card(a,28,11.3,11,7.4,'Test  540',size=6.8)
    label(a,46.7,41.5,'Augmented run only',color=ORANGE,size=6.5,bold=True)
    aug=[ImageOps.mirror(ims[0]),ims[0].rotate(20),ImageEnhance.Brightness(ims[0]).enhance(.8)]
    for i,im in enumerate(aug):photo(a,im,41.7+i*4.2,32,3.7,5.6)
    arrow(a,[(39.3,35),(41.2,35)],ORANGE)
    for x,t in zip([43.55,47.75,51.95],['Flip','Rotate','Color']):label(a,x,30.4,t,size=5.8,color=ORANGE)
    arrow(a,[(39.2,33),(40,33),(40,28),(43,28)],color='#657B88',dashed=True,head=4)
    label(a,48,28,'Base: no aug.',size=6.0,color='#657B88')
    arrow(a,[(53,28),(56,28),(56,25),(59,25)],color='#657B88',dashed=True,head=4)
    arrow(a,[(54.2,35),(56.7,35),(56.7,25),(59,25)],ORANGE)
    arrow(a,[(39.3,25),(59,25)])
    arrow(a,[(39.3,15),(55.2,15),(55.2,25),(59,25)])
    label(a,47.3,10,'No random aug.',size=6.5,color='#657B88')
    card(a,59,20,9,10,'ImageNet\nnormalize',size=6.4)
    arrow(a,[(68.3,25),(71,25)])
    tensor(a,75.8,25,6.6,9.4,4)
    label(a,76.5,37,'ResNet18',bold=True,size=7.2)
    label(a,76.5,13.7,'ImageNet init.\nFull fine-tuning',size=6.3)
    arrow(a,[(81,25),(83.5,25)])
    card(a,83.5,20,7.8,10,'GAP\nFC 7',size=6.8)
    label(a,87.4,17.3,'7 logits',size=6.5)
    for i in range(7):a.add_patch(Rectangle((94,20.5+i*1.35),3.5,1.05,fc='#DDE8EE',ec=BLUE,lw=.35))
    arrow(a,[(91.6,25),(93.5,25)])
    label(a,95.7,15.6,'Argmax',size=6.5)
    label(a,95.7,12.3,'Class ID',size=6.5,bold=True)
    label(a,91,45,'Training only',color=ORANGE,size=6.5,bold=True)
    card(a,83.1,34.8,15.6,7.5,'CE(logits, y)\nAdamW / cosine',fill='#F9ECDD',color=ORANGE,size=6.1)
    arrow(a,[(87.4,30.3),(87.4,34.5)],ORANGE,True)
    label(a,60.5,3.4,'Val macro-F1  →  selected checkpoint  →  held-out test',size=6.8,color='#536B7B')

    label(b,0,53,'b  ResNet18 feature hierarchy',size=8.5,bold=True,ha='left',va='top')
    centers=[9,28.5,48,67.5,87]
    titles=['Stem','Layer 1','Layer 2','Layer 3','Layer 4']
    shapes=['64 x 32 x 32','64 x 32 x 32','128 x 16 x 16','256 x 8 x 8','512 x 4 x 4']
    hs=[26,26,19,13.5,9];ws=[11,11,9,7,5];ds=[3,3,4,5,7]
    for i,(cx,h,w,d) in enumerate(zip(centers,hs,ws,ds)):
        label(b,cx,43.8,titles[i],bold=True,size=7)
        tensor(b,cx,26,w,h,d)
        label(b,cx,7.9,shapes[i],size=6.4)
        if i<4:arrow(b,[(cx+w/2+(d-1)*.55+1.5,26),(centers[i+1]-ws[i+1]/2-1,26)])
    label(b,50,1,'C x H x W; stacked planes are schematic, not activation maps',size=6.3,color='#657B88')
    # Dedicated same-shape block: addition has two distinct input ports.
    label(c,0,98.1,'c  Residual block detail',size=8.5,bold=True,ha='left',va='top')
    label(c,62,86,'x',size=8,bold=True)
    card(c,36,62,53,14,'Conv 3x3\nBN + ReLU',fill='#E8F1E9',size=7)
    card(c,36,36,53,14,'Conv 3x3\nBN',fill='#E8F1E9',size=7)
    arrow(c,[(62.5,82),(62.5,76.7)])
    arrow(c,[(62.5,61.5),(62.5,50.6)])
    c.add_patch(Circle((62.5,23),6.2,fc='white',ec=TEAL,lw=.8,zorder=4));label(c,62.5,23,'+',size=9)
    arrow(c,[(62.5,35.6),(62.5,29.7)])
    arrow(c,[(62.5,81),(21,81),(21,23),(55.8,23)],TEAL)
    label(c,9.5,53,'Identity\nshortcut',size=6.2,color=TEAL)
    card(c,80,16,19,14,'ReLU',fill='#F9ECDD',size=6.8)
    arrow(c,[(69.1,23),(79.4,23)])
    label(c,50,5.5,'Same-shape block shown; stage entries use\n1x1 / stride-2 projection (Layers 2-4).',size=6.1,color='#657B88')

    fig.canvas.draw();out=ROOT/'figures';qa=ROOT/'qa';out.mkdir(exist_ok=True);qa.mkdir(exist_ok=True)
    if require_matplotlib_panel_alignment:
        require_matplotlib_panel_alignment(fig,json_out=qa/'model_overview.alignment.json',strict=True)
    else:
        (qa/'model_overview.alignment.json').write_text(json.dumps({'status':'unavailable','reason':'external geometry auditor not installed'}))
    fig.savefig(out/'model_overview.pdf')
    fig.savefig(out/'model_overview.svg')
    fig.savefig(out/'model_overview.png',dpi=600)
    fig.savefig(out/'model_overview.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    # A light preview is convenient for inline review; PDF/SVG remain authoritative.
    fig.savefig(out/'model_overview_preview.png',dpi=300)
    plt.close(fig)

if __name__=='__main__':main()
