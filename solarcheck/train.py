"""One predeclared CNN experiment, validation selection, then final test evaluation."""
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix,precision_score,recall_score,f1_score
from PIL import Image
from .model import DefectCNN,preprocess
ROOT=Path(__file__).resolve().parents[1]
SEED=42

def split_indices(labels,types):
    idx=np.arange(len(labels));strata=np.array([str(y)+'_'+str(t) for y,t in zip(labels,types)])
    train,rest=train_test_split(idx,test_size=.4,random_state=SEED,stratify=strata)
    val,test=train_test_split(rest,test_size=.5,random_state=SEED,stratify=strata[rest])
    return train,val,test

def threshold_for_recall(y,p,target=.9):
    choices=np.unique(np.r_[0.,p])
    feasible=[float(t) for t in choices if recall_score(y,p>=t,zero_division=0)>=target]
    return max(feasible)

def metrics(y,p,t):
    pred=p>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'average_precision':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),'recall':float(recall_score(y,pred,zero_division=0)),'precision':float(precision_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'false_positive_rate':float(fp/(fp+tn)),'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]],'threshold':float(t)}

def main(epochs=12):
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
    # CPU chosen for repeatable portable training; Apple GPU is optional future work.
    device=torch.device('cpu')
    raw=np.load(ROOT/'data/raw.npz');images,proba,types=raw['images'],raw['proba'],raw['types']
    selected=np.where((proba==0)|(proba==1))[0]
    hashes=[hashlib.sha256(images[i].tobytes()).hexdigest() for i in selected]
    seen=set();keep=[]
    for i,h in zip(selected,hashes):
        if h not in seen:keep.append(i);seen.add(h)
    ids=np.array(keep);y=proba[ids].astype(int);types=types[ids];images=images[ids]
    train,val,test=split_indices(y,types)
    splits=np.full(len(y),'train',dtype='<U5');splits[val]='val';splits[test]='test'
    manifest=pd.DataFrame({'source_index':ids,'label':y,'cell_type':types,'split':splits,'sha256':[hashlib.sha256(im.tobytes()).hexdigest() for im in images]})
    manifest.to_csv(ROOT/'reports/split_manifest.csv',index=False)
    x=torch.stack([preprocess(Image.fromarray(im)) for im in images]);yt=torch.tensor(y,dtype=torch.float32)
    # Baseline uses downsampled grayscale intensities, no learned visual features.
    xb=np.stack([np.asarray(Image.fromarray(im).resize((16,16)),dtype=float).ravel()/255 for im in images])
    baseline=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,class_weight='balanced',random_state=SEED))
    baseline.fit(xb[train],y[train]);bp=baseline.predict_proba(xb[val])[:,1];bt=threshold_for_recall(y[val],bp)
    model=DefectCNN().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y[train]==0).sum()/(y[train]==1).sum()],dtype=torch.float32))
    loader=DataLoader(TensorDataset(x[train],yt[train]),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    history=[];best=-1;start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();loss_sum=0
        for batch,target in loader:
            # Cell orientations may vary; apply identical intensity scale at inference.
            if torch.rand(())<.5:batch=torch.flip(batch,[-1])
            if torch.rand(())<.5:batch=torch.flip(batch,[-2])
            optimizer.zero_grad();loss=loss_fn(model(batch),target);loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(batch)
        model.eval()
        with torch.inference_mode():vp=torch.sigmoid(model(x[val])).numpy()
        ap=float(average_precision_score(y[val],vp));row={'epoch':epoch,'train_loss':loss_sum/len(train),'val_average_precision':ap};history.append(row);print(row,flush=True)
        if ap>best:
            best=ap
            torch.save({'state_dict':model.state_dict(),'epoch':epoch,'seed':SEED,'size':96},ROOT/'artifacts/model.pt')
    checkpoint=torch.load(ROOT/'artifacts/model.pt',weights_only=True);model.load_state_dict(checkpoint['state_dict']);model.eval()
    with torch.inference_mode():vp=torch.sigmoid(model(x[val])).numpy();tp=torch.sigmoid(model(x[test])).numpy()
    threshold=threshold_for_recall(y[val],vp);checkpoint['threshold']=threshold;torch.save(checkpoint,ROOT/'artifacts/model.pt')
    btest=baseline.predict_proba(xb[test])[:,1]
    report={'seed':SEED,'device':str(device),'epochs':epochs,'selected_epoch':checkpoint['epoch'],'training_seconds':time.perf_counter()-start,'source_images':len(proba),'excluded_ambiguous':int(((proba!=0)&(proba!=1)).sum()),'exact_duplicates_removed':len(selected)-len(ids),'counts':{s:int((splits==s).sum()) for s in ['train','val','test']},'threshold_rule':'Highest validation threshold reaching at least 90% recall; test performance may differ.','selection_rule':f'Maximum validation average precision across {epochs} fixed epochs. Test not used for selection.','cnn_validation':metrics(y[val],vp,threshold),'cnn_test':metrics(y[test],tp,threshold),'baseline_test':metrics(y[test],btest,bt),'test_positive_prevalence':float(y[test].mean()),'test_slices':{str(t):metrics(y[test][types[test]==t],tp[types[test]==t],threshold) for t in np.unique(types)},'limitations':['Image-level split, not module-level: module IDs are unavailable in this dataset interface. Related cells can cross splits; results do not establish unseen-module generalization.','Ambiguous labels excluded: binary extreme-label task is easier and not the original four-level benchmark.','Scores are not calibrated probabilities.','One seed, one small architecture, no field validation.'],'versions':{'torch':str(torch.__version__)}}
    (ROOT/'reports/metrics.json').write_text(json.dumps(report,indent=2));(ROOT/'reports/history.json').write_text(json.dumps(history,indent=2))
    pd.DataFrame({'source_index':ids[test],'label':y[test],'cell_type':types[test],'score':tp,'flagged':tp>=threshold}).to_csv(ROOT/'reports/test_predictions.csv',index=False)
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=12);main(parser.parse_args().epochs)
