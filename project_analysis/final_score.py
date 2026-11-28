"""Preflight all frozen pipelines before the one-time December scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT))
from preprocessing import FEATURES


def predict(model_path, inputs, output):
    model=joblib.load(model_path)
    frame=pd.read_csv(inputs,usecols=FEATURES)
    prob=model.predict_proba(frame[FEATURES])[:,list(model.classes_).index(1)]
    assert len(prob)==len(frame) and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    np.save(output,prob)


def run(legacy):
    models=sorted((ROOT/'results').glob('*/model.joblib'))
    assert len(models)==11
    lock=HERE/'december_freeze.json'
    if lock.exists():
        raise SystemExit('December scoring already initiated. See december_freeze.json.')
    with TemporaryDirectory() as temp:
        path=Path(temp)
        inputs=path/'inputs.csv'
        pd.read_csv(ROOT/'final_csv/validation_T2h.csv',nrows=32,usecols=FEATURES).to_csv(inputs,index=False)
        commands={}
        manifest={}
        for model in models:
            name=model.parent.name
            versions=json.loads((model.parent/'settings.json').read_text())['package_versions']
            python=legacy if versions['scikit-learn']=='1.6.1' else sys.executable
            commands[name]=[python,'-I',str(Path(__file__).resolve()),'--predict',str(model),str(inputs),str(path/f'{name}.npy')]
            subprocess.run(commands[name],cwd=path,check=True)
            manifest[name]=hashlib.file_digest(model.open('rb'),'sha256').hexdigest()
        with lock.open('x') as f:
            json.dump({'models_sha256':manifest,'features':FEATURES,'metrics':['roc_auc','average_precision'],'period':'2025-12-01 to 2025-12-31'},f,indent=2)
        data=pd.read_csv(ROOT/'final_csv/test_T2h.csv',usecols=FEATURES+['LATE','row_id','FlightDate'])
        assert len(data)==68146 and data.row_id.is_unique and data.FlightDate.between('2025-12-01','2025-12-31').all()
        data[FEATURES].to_csv(inputs,index=False)
        results=[]
        for name,cmd in commands.items():
            subprocess.run(cmd,cwd=path,check=True)
            prob=np.load(path/f'{name}.npy')
            results.append([name,len(data),data.LATE.mean(),roc_auc_score(data.LATE,prob),average_precision_score(data.LATE,prob)])
        pd.DataFrame(results,columns=['model','flights','late_rate','roc_auc','average_precision']).to_csv(HERE/'december_final.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--legacy-python')
    parser.add_argument('--predict',nargs=3)
    args=parser.parse_args()
    if args.predict:
        predict(*args.predict)
    else:
        run(args.legacy_python)
