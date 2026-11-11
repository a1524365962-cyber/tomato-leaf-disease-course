"""Post-hoc true-class Grad-CAM from the actual validation-selected weights."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','1')
import json,tarfile
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import interpolate
from PIL import Image,ImageFilter,ImageEnhance
from torchvision import transforms
import train_experiment as study
R=Path(__file__).resolve().parent
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
cases=json.loads((R/'visual_examples.json').read_text())['test_cases'][:4]
testrows=[r for r in json.loads((R/'manifest.json').read_text())['rows'] if r['split']=='test']
tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
out={};meta=[]
for j,case in enumerate(cases):
    im=Image.open(R/case['path']).convert('RGB').resize((128,128),Image.Resampling.BILINEAR)
    if case['condition']=='dark': im=ImageEnhance.Brightness(im).enhance(.6)
    elif case['condition']=='blur':im=im.filter(ImageFilter.GaussianBlur(1.5))
    out[f'input_{j}']=np.array(im)
    for method,key in [('resnet18','baseline'),('resnet18_aug','augmented')]:
        ck=torch.load(R/'runs'/f'{method}_seed2026'/'best.pt',map_location='cpu');study.CLASSES=ck['config']['classes']
        m=study.make_model(method,False);m.load_state_dict(ck['model']);m=m.cuda().eval();features=[]
        hook=m.layer4.register_forward_hook(lambda module,inp,value:features.append(value))
        idx=next(i for i,r in enumerate(testrows) if r['path']==case['path']);start=idx//64*64;local=idx-start
        ds=study.Leaves(testrows[start:start+64],condition=case['condition'])
        logits=m(torch.stack([ds[i][0] for i in range(len(ds))]).cuda());target=study.CLASSES.index(case['target'])
        assert study.CLASSES[int(logits[local].argmax())]==case[key], (j,key)
        case[key+'_cam_forward']=study.CLASSES[int(logits[local].argmax())]
        g=torch.autograd.grad(logits[local,target],features[0])[0][local:local+1]
        feature=features[0][local:local+1]
        cam=(g.mean((2,3),keepdim=True)*feature).sum(1,keepdim=True).relu()
        cam=interpolate(cam,size=(128,128),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy()
        cam=(cam-cam.min())/max(float(cam.max()-cam.min()),1e-12)
        out[f'{key}_{j}']=cam;hook.remove();del m,logits,features,g
    meta.append(case)
np.savez_compressed(R/'cams.npz',**out)
(R/'cams_metadata.json').write_text(json.dumps({'cases':meta,'target':'ground-truth class for both models, post-hoc analysis only','layer':'layer4 (4x4 at input 128)','normalization':'per-image min-max, not comparable magnitude across images','seed':2026},indent=2))
print('computed',len(meta),'paired CAMs')
