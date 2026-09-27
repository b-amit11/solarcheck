import numpy as np
import torch
from torch import nn
from PIL import Image,ImageOps
SIZE=96
class DefectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.AdaptiveAvgPool2d((4,4)))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(64*4*4,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,1))
    def forward(self,x):return self.head(self.features(x)).squeeze(1)

def preprocess(image):
    image=ImageOps.exif_transpose(image).convert('L').resize((SIZE,SIZE),Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image,dtype=np.float32).copy()/255)[None]

def load_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=DefectCNN();model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,checkpoint

def predict(model,image):
    with torch.inference_mode():return float(torch.sigmoid(model(preprocess(image)[None]))[0])
