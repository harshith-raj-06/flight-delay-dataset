"""Locked, seed-11 XGBoost experiments on development exports only."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from preprocessing import FEATURES, SCHEDULE, ROUTE, WEATHER, NativeInputs, development_folds


def run():
    rows, effects, dependence = [], [], []
    reference = joblib.load(ROOT/'results/xgboost/model.joblib')['model']
    saved = pd.read_parquet(ROOT/'results/xgboost/predictions.parquet').query('seed == 11').set_index('row_id')
    for horizon in [2, 6]:
        dev = pd.concat([pd.read_csv(ROOT/f'final_csv/{split}_T{horizon}h.csv') for split in ['train','validation']])
        dev.columns = [c.replace('_T6h','_T2h') for c in dev]
        dev = dev.set_index('row_id',verify_integrity=True).sort_values('checkpoint_utc',kind='stable')
        for fold,train,val in development_folds(dev):
            if fold < 3:
                continue
            prep = NativeInputs('category').fit(train[FEATURES])
            X, V = prep.transform(train[FEATURES]), prep.transform(val[FEATURES])
            groups = {'ABCD':FEATURES} if horizon == 6 else {'A':SCHEDULE,'AB':SCHEDULE+ROUTE,'ABC':SCHEDULE+ROUTE+WEATHER,'ABCD':FEATURES}
            for group,cols in groups.items():
                model = clone(reference).set_params(random_state=11,n_jobs=5)
                model.fit(X[cols],train.LATE)
                prob = pd.Series(model.predict_proba(V[cols])[:,1],index=val.index)
                if horizon==2 and group=='ABCD':
                    # The stored run sorted by UTC; both fits must reproduce its ranking.
                    assert abs(average_precision_score(val.LATE,prob)-average_precision_score(val.LATE,saved.loc[val.index,'p_late'])) < .001
                for airport,subset in [('All',val),*val.groupby('Origin')]:
                    y,p = subset.LATE,prob.loc[subset.index]
                    rows.append(dict(hours=horizon,group=group,fold=fold,airport=airport,flights=len(y),ap=average_precision_score(y,p),auc=roc_auc_score(y,p),lift=average_precision_score(y,p)/y.mean()))
                print(horizon,fold,group,flush=True)
                if (horizon,fold,group)!=(2,4,'ABCD'):
                    continue
                sample=V.sample(1500,random_state=11)
                target=val.loc[sample.index,'LATE']
                base=average_precision_score(target,model.predict_proba(sample)[:,1])
                contrib=model.get_booster().predict(xgb.DMatrix(sample,enable_categorical=True),pred_contribs=True)
                rng=np.random.default_rng(11)
                for j,col in enumerate(cols):
                    shuffled=sample.copy()
                    shuffled[col]=sample[col].array.take(rng.permutation(len(sample)))
                    ap=average_precision_score(target,model.predict_proba(shuffled)[:,1])
                    effects.append(dict(feature=col,mean_abs_shap=float(np.abs(contrib[:,j]).mean()),permutation_ap_drop=base-ap))
                for col in ['late_rate_60min_T2h','dep_hour','wx_vsby_T2h','n_departed_60min_T2h']:
                    for v in np.unique(sample[col].dropna().quantile([.05,.25,.5,.75,.95])):
                        modified=sample.copy();modified[col]=v
                        dependence.append(dict(feature=col,value=v,mean_predicted_risk=float(model.predict_proba(modified)[:,1].mean())))
    out=ROOT/'site/data/studies.json'
    temp=out.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(scores=rows,effects=effects,dependence=dependence),allow_nan=False))
    temp.replace(out)


if __name__=='__main__':
    run()
