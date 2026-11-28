"""Analyze held-out development predictions. Run from any directory."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SEEDS = {11, 23, 42}


def analyze():
    meta = pd.read_csv(ROOT/'final_csv/validation_T2h.csv', usecols=['row_id','Origin','FlightDate','LATE','dep_hour'])
    meta = meta[meta.FlightDate >= '2025-10-13'].set_index('row_id', verify_integrity=True)
    benchmark, reliability, stability, errors = [], [], [], []
    for file in sorted((ROOT/'results').glob('*/predictions.parquet')):
        name = file.parent.name
        preds = pd.read_parquet(file)
        assert set(preds.seed) == SEEDS and set(preds.fold) == {3,4}
        scores = []
        for seed, rows in preds.groupby('seed'):
            rows = rows.set_index('row_id',verify_integrity=True).sort_index()
            assert rows.index.equals(meta.sort_index().index)
            data = rows.join(meta.drop(columns='LATE'))
            assert (rows.LATE == meta.loc[rows.index,'LATE']).all()
            assert rows.p_late.between(0,1).all() and ((data.fold == 3) == (data.FlightDate <= '2025-11-02')).all()
            for fold,g in data.groupby('fold'):
                scores.append((roc_auc_score(g.LATE,g.p_late),average_precision_score(g.LATE,g.p_late)))
                if seed != 11:
                    continue
                for b,v in g.groupby(pd.cut(g.p_late,np.linspace(0,1,11),include_lowest=True),observed=True):
                    reliability.append([name,fold,str(b),len(v),v.p_late.mean(),v.LATE.mean(),brier_score_loss(v.LATE,v.p_late)])
                for airport,v in g.groupby('Origin'):
                    ap = average_precision_score(v.LATE,v.p_late)
                    stability.append([name,fold,airport,len(v),v.LATE.mean(),ap,ap/v.LATE.mean(),roc_auc_score(v.LATE,v.p_late)])
                # Fixed 10% review capacity; fractional allocation at a tied cutoff.
                k = int(np.ceil(len(g)*0.1))
                cutoff = g.p_late.nlargest(k).iloc[-1]
                above = g.p_late.gt(cutoff)
                tied = g.p_late.eq(cutoff)
                alert = above.astype(float) + tied * ((k-above.sum())/tied.sum())
                for period,v in [('All',g),*g.groupby(pd.cut(g.dep_hour,[-1,5,11,17,23],labels=['00–05','06–11','12–17','18–23']),observed=True)]:
                    a = alert.loc[v.index]
                    tp=(a*v.LATE).sum(); fp=(a*(1-v.LATE)).sum()
                    errors.append([name,fold,period,len(v),tp,fp,v.LATE.sum()-tp,tp/v.LATE.sum(),tp/a.sum() if a.sum() else np.nan])
        scores=np.array(scores)
        benchmark.append([name,*scores.mean(axis=0),*scores.std(axis=0,ddof=1)])
    for data,cols,file in [
        (benchmark,'model auc ap auc_sd ap_sd','benchmark'),
        (reliability,'model fold bin flights predicted observed brier','calibration'),
        (stability,'model fold airport flights late_rate ap lift auc','robustness'),
        (errors,'model fold hours flights true_alert false_alert missed recall precision','errors')]:
        pd.DataFrame(data,columns=cols.split()).to_csv(OUT/f'{file}.csv',index=False)


if __name__=='__main__':
    analyze()
