import json,hashlib
from pathlib import Path
from importlib.metadata import version
import numpy as np
from elpv_dataset.utils import load_dataset
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 images,proba,types=load_dataset();(ROOT/'data').mkdir(exist_ok=True)
 np.savez_compressed(ROOT/'data/raw.npz',images=images,proba=proba,types=types)
 (ROOT/'reports/source.json').write_text(json.dumps({'source':'https://github.com/zae-bayern/elpv-dataset','package_version':version('elpv-dataset'),'license_images':'CC-BY-NC-SA-4.0','image_shape':list(images.shape),'labels':{str(p):int((proba==p).sum()) for p in np.unique(proba)},'archive_sha256':hashlib.sha256((ROOT/'data/raw.npz').read_bytes()).hexdigest()},indent=2))
 print('Loaded',images.shape,'label values',np.unique(proba))
