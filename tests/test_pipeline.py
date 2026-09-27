import numpy as np
import torch
from PIL import Image
from solarcheck.model import DefectCNN,preprocess,predict
from solarcheck.train import split_indices,threshold_for_recall

def test_split_disjoint_reproducible():
 y=np.array([0,1]*100);types=np.array(['mono']*100+['poly']*100)
 a,b,c=split_indices(y,types)
 assert not(set(a)&set(b) or set(a)&set(c) or set(b)&set(c))
 assert set(a)|set(b)|set(c)==set(range(200))
 assert all(np.array_equal(x,z) for x,z in zip((a,b,c),split_indices(y,types)))

def test_preprocess_rgb_and_grayscale_match():
 gray=Image.new('L',(300,300),128)
 assert torch.equal(preprocess(gray),preprocess(gray.convert('RGB')))
 assert preprocess(gray).shape==(1,96,96)
 assert torch.isfinite(preprocess(gray)).all()

def test_gradients_and_weight_update():
 torch.manual_seed(7);m=DefectCNN();x=torch.rand(4,1,96,96);y=torch.tensor([0.,1.,0.,1.])
 old=m.features[0].weight.detach().clone();optimizer=torch.optim.Adam(m.parameters())
 loss=torch.nn.functional.binary_cross_entropy_with_logits(m(x),y);loss.backward()
 assert torch.isfinite(m.features[0].weight.grad).all()
 optimizer.step();assert not torch.equal(old,m.features[0].weight)

def test_threshold_respects_validation_recall():
 y=np.array([0,0,1,1]);p=np.array([.1,.8,.6,.9]);t=threshold_for_recall(y,p,1.)
 assert t==.6 and ((p[y==1]>=t).mean()==1)

def test_model_serialization(tmp_path):
 from solarcheck.model import load_model
 m=DefectCNN().eval();path=tmp_path/'m.pt';torch.save({'state_dict':m.state_dict(),'threshold':.5},path)
 loaded,_=load_model(path);im=Image.new('L',(300,300),50)
 assert predict(m,im)==predict(loaded,im)
