from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_app_loads_checkpoint_and_report():
 app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
 assert not app.exception
 assert len(app.metric)==1
 assert app.metric[0].label=='Defect score'
