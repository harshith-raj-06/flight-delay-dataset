"""Publish saved development outputs and the aggregate final scoreboard."""
import csv
import json
from pathlib import Path

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
RESULTS = ROOT / 'results'
ANALYSIS = ROOT / 'project_analysis'


def rows(path):
    if path.name.startswith('test_') or 'final_csv' in path.parts:
        raise ValueError('Raw flight exports are outside the site build')
    print('Reading', path.relative_to(ROOT))
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write(name, content):
    (SITE / 'data' / (name+'.json')).write_text(json.dumps(content, separators=(',', ':')))


benchmark = rows(ANALYSIS/'benchmark.csv')
folds = {r['model']:rows(RESULTS/r['model']/'evaluation.csv') for r in benchmark}
for r in benchmark:
    ev=folds[r['model']]
    r['lift']=sum(float(v['ap'])/float(v['late_rate']) for v in ev)/len(ev)
write('tuning',{r['model']:rows(RESULTS/r['model']/'tuning.csv') for r in benchmark})
write('models',dict(summary=benchmark,folds=folds,curves={p.stem:rows(p) for p in [
    RESULTS/'decision_tree/depth_curve.csv',RESULTS/'decision_tree/pruning_curve.csv',
    RESULTS/'pca_logistic_regression/pca_components_per_fold.csv',RESULTS/'pca_logistic_regression/pca_variance.csv']}))
write('errors',{k:rows(ANALYSIS/(k+'.csv')) for k in ['robustness','errors','calibration']})
write('interpretation',{p.parent.name:rows(p) for p in RESULTS.glob('*/feature_importances.csv')})
audit=[l.strip('|').split('|') for l in (ROOT/'COLUMN_AUDIT.md').read_text().splitlines() if l.startswith('|')][2:]
write('protocol',[dict(zip(['Input','Availability','Definition'],map(str.strip,r))) for r in audit])
if (ANALYSIS/'december_final.csv').exists():
    write('test',rows(ANALYSIS/'december_final.csv'))
else:
    (SITE/'data/test.json').unlink(missing_ok=True)

# Only development exports supply the descriptive charts.
import pandas as pd
development = pd.concat([
    pd.read_csv(ROOT/'final_csv'/filename, usecols=['month','dow','dep_hour','Origin','Reporting_Airline','LATE','wx_vsby_T2h','wx_sknt_T2h','wx_skyl1_T2h','wx_ts_T2h','wx_fg_T2h','wx_sn_T2h','late_rate_60min_T2h','n_departed_60min_T2h','n_scheduled_next_60min_T2h'])
    for filename in ('train_T2h.csv', 'validation_T2h.csv')
], ignore_index=True)
def rate_rows(columns):
    return development.groupby(columns).LATE.agg(flights='size', late_rate='mean').reset_index().to_dict('records')
patterns = {}
for column, label, bins in [
    ('wx_vsby_T2h','Visibility (miles)',[0,1,3,6,10,float('inf')]),
    ('wx_sknt_T2h','Wind speed (knots)',[0,5,10,20,float('inf')]),
    ('wx_skyl1_T2h','Cloud base (feet)',[0,1000,3000,10000,float('inf')]),
    ('late_rate_60min_T2h','Prior-hour late fraction',[0,.1,.25,.5,1]),
    ('n_departed_60min_T2h','Prior-hour departures',[0,10,30,60,float('inf')]),
    ('n_scheduled_next_60min_T2h','Next-hour scheduled departures',[0,10,30,60,float('inf')]),
]:
    # Label lower edge explicitly; the first interval includes zero.
    labels = [f'{a:g}–{b:g}' if i==0 else f'>{a:g}–{b:g}' if b!=float('inf') else f'>{a:g}' for i,(a,b) in enumerate(zip(bins,bins[1:]))]
    development[label]=pd.cut(development[column],bins,labels=labels,include_lowest=True).astype('string').fillna('Unknown')
    patterns[label]=rate_rows([label])
for column, label in [('wx_ts_T2h','Thunderstorm'),('wx_fg_T2h','Fog'),('wx_sn_T2h','Snow')]:
    development[label]=development[column].map({0:'Absent in known codes',1:'Reported'}).fillna('Unknown')
    patterns[label]=rate_rows([label])
write('eda', {**patterns, 'month':rate_rows(['month']), 'hour_weekday':rate_rows(['dow','dep_hour']), 'airport':rate_rows(['Origin']), 'carrier':rate_rows(['Reporting_Airline']), 'hour':rate_rows(['dep_hour']), 'weekday':rate_rows(['dow'])})

from sklearn.metrics import precision_recall_curve, roc_curve
import numpy as np
classification_curves = {}
for entry in benchmark:
    key = entry['model']
    predictions = pd.read_parquet(RESULTS/key/'predictions.parquet', filters=[('seed','=',11)])
    classification_curves[key] = {}
    for fold, block in predictions.groupby('fold'):
        fpr, tpr, _ = roc_curve(block.LATE,block.p_late)
        precision, recall, _ = precision_recall_curve(block.LATE,block.p_late)
        def points(xs,ys,xname,yname):
            indices = np.unique(np.linspace(0,len(xs)-1,min(130,len(xs))).astype(int))
            return [{xname:float(xs[i]),yname:float(ys[i])} for i in indices]
        classification_curves[key][str(fold)] = dict(
            roc=points(fpr,tpr,'false_positive_rate','recall'),
            pr=points(recall[::-1],precision[::-1],'recall','precision'),
            prevalence=float(block.LATE.mean()))
write('classification',classification_curves)
