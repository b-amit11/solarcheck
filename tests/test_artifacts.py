from pathlib import Path
import json
import pandas as pd
import pytest
from PIL import Image
from solarcheck.model import load_model,predict
ROOT=Path(__file__).resolve().parents[1]

def test_saved_predictions_and_split_integrity():
 manifest=pd.read_csv(ROOT/'reports/split_manifest.csv');pred=pd.read_csv(ROOT/'reports/test_predictions.csv')
 assert not manifest.sha256.duplicated().any()
 assert set(pred.source_index)==set(manifest[manifest.split=='test'].source_index)
 model,checkpoint=load_model(ROOT/'artifacts/model.pt')
 for path in (ROOT/'examples').glob('*.png'):
  idx=int(path.name.split('_')[0]);expected=float(pred.loc[pred.source_index==idx,'score'].iloc[0])
  assert predict(model,Image.open(path))==pytest.approx(expected,abs=1e-6)
 report=json.loads((ROOT/'reports/metrics.json').read_text())
 assert checkpoint['threshold']==report['cnn_test']['threshold']
