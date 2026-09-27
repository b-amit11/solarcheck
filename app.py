import json
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image,UnidentifiedImageError
from solarcheck.model import load_model,predict
ROOT=Path(__file__).parent
st.set_page_config(page_title='SolarCheck',page_icon='☀',layout='wide')
st.title('SolarCheck')
st.write('A PyTorch research prototype for reviewing potentially defective solar cells in electroluminescence imagery.')
st.warning('Educational inspection aid. Not validated on unseen solar modules or ordinary phone photographs. A low score does not certify a cell as defect-free.')
if not (ROOT/'artifacts/model.pt').exists():
 st.info('Run the preparation and training commands in README.md first.');st.stop()
@st.cache_resource
def model():return load_model(ROOT/'artifacts/model.pt')
m,checkpoint=model()
report=json.loads((ROOT/'reports/metrics.json').read_text())
inspect,results,method=st.tabs(['Inspect a cell','Evaluation','Training and limitations'])
with inspect:
 examples=sorted((ROOT/'examples').glob('*.png'))
 source=st.radio('Image source',['Dataset example','Upload an EL cell image'],horizontal=True)
 image=None
 if source=='Dataset example' and examples:
  chosen=st.selectbox('Held-out example',[p.name for p in examples]);image=Image.open(ROOT/'examples'/chosen)
  st.caption('Examples are from the final test split. Their names include source labels; these labels are not inputs to the model.')
 elif source!='Dataset example':
  upload=st.file_uploader('Single cell crop; PNG or JPEG, at most 5 MB',type=['png','jpg','jpeg'])
  if upload:
   try:
    if upload.size>5_000_000:raise ValueError('File exceeds 5 MB.')
    image=Image.open(upload)
    if image.width*image.height>16_000_000:raise ValueError('Image exceeds 16 megapixels.')
    image.load()
   except (ValueError,UnidentifiedImageError,OSError,Image.DecompressionBombError) as e:st.error(str(e));image=None
 if image is not None:
  a,b=st.columns(2)
  with a:st.image(image,width=300)
  with b:
   score=predict(m,image);threshold=checkpoint['threshold'];st.metric('Defect score',f'{score:.3f}')
   st.write('**Flag for review**' if score>=threshold else '**Below review threshold**')
   st.caption(f'Fixed validation-selected threshold: {threshold:.3f}. Score is not a calibrated probability.')
 st.caption('Dataset images: ELPV, Buerhop-Lutz et al. and Deitsch et al.; CC BY-NC-SA 4.0. Examples are converted to PNG for display. See THIRD_PARTY.md.')
with results:
 st.subheader('Final test results')
 st.dataframe(pd.DataFrame({'CNN':report['cnn_test'],'Pixel logistic baseline':report['baseline_test']}).drop(index=['confusion_matrix']),width='stretch')
 st.write('CNN confusion matrix: rows = actual normal/defective; columns = predicted normal/defective.')
 st.dataframe(pd.DataFrame(report['cnn_test']['confusion_matrix'],index=['Actual normal','Actual defective'],columns=['Below threshold','Flagged']))
 st.write('Test class prevalence:',round(report['test_positive_prevalence'],3))
 st.line_chart(pd.DataFrame(json.loads((ROOT/'reports/history.json').read_text())).set_index('epoch')[['val_average_precision']])
 st.json(report['test_slices'])
with method:
 st.write('Three convolution blocks, ReLU, pooling, a dense head, and dropout. Trained from scratch with Adam, weighted binary cross-entropy, and random flips. Checkpoint chosen using validation average precision; threshold chosen for at least 90% validation recall.')
 st.write('Only labels 0 and 1 are used. Intermediate labels are excluded. The 60/20/20 split is stratified by label and cell type; exact duplicate images are removed before splitting.')
 for item in report['limitations']:st.write('• '+item)
 st.write('No cloud model API, pretrained weights, or automated quality certification.')
