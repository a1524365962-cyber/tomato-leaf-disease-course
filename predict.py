"""Single-image inference from a trusted project checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import train_experiment as study

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--image',type=Path,required=True); args=p.parse_args()
    ck=torch.load(args.checkpoint,map_location='cpu'); study.CLASSES=ck['config']['classes']
    model=study.make_model(ck['config']['method'],pretrained=False); model.load_state_dict(ck['model']); model.eval()
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    with Image.open(args.image) as im: x=tf(im.convert('RGB').resize((128,128),Image.Resampling.BILINEAR)).unsqueeze(0)
    with torch.no_grad(): probs=model(x).softmax(1)[0]
    print(json.dumps([{'class':study.CLASSES[int(i)],'softmax_probability':float(probs[i])} for i in probs.argsort(descending=True)[:3]],indent=2))
    print('Closed-set prediction only; softmax values are not calibrated diagnostic confidence.')
if __name__=='__main__': main()
