"""Render the benchmark and the three rubric activities from development outputs."""
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

root=Path(__file__).resolve().parent
names={'adaboost':'AdaBoost','catboost':'CatBoost','decision_tree':'Decision tree','gradient_boosting':'Gradient boosting','histgb':'Histogram GB','lightgbm':'LightGBM','logistic_regression':'Logistic regression','majority':'Majority prior','pca_logistic_regression':'PCA + logistic','random_forest':'Random forest','xgboost':'XGBoost'}
def load(name):
    frame=pd.read_csv(root/f'{name}.csv')
    frame['model']=frame.model.map(names)
    return frame

def page(title, subtitle, note):
    fig=plt.figure(figsize=(11,8.5))
    fig.text(.07,.92,title,fontsize=23,weight='bold',color='#124851')
    fig.text(.07,.86,subtitle,fontsize=12)
    fig.text(.07,.07,textwrap.fill(note,112),fontsize=11,linespacing=1.4)
    return fig

with PdfPages(root/'Analysis_and_benchmark.pdf') as pdf:
    b=load('benchmark').sort_values('ap',ascending=False)
    fig=page('Two-hour departure-delay benchmark','2025 development evaluation  /  Atlanta, Chicago O’Hare and Los Angeles','XGBoost and LightGBM differed by 0.0011 mean AP. Two reporting periods provide limited evidence for a stable model ranking. PCA retained 31 components on both reporting folds. Plain logistic regression scored higher on both metrics.')
    ax=fig.add_axes([.07,.25,.86,.55]);ax.axis('off')
    rows=[[r.model,f'{r.auc:.4f} ± {r.auc_sd:.4f}',f'{r.ap:.4f} ± {r.ap_sd:.4f}'] for r in b.itertuples()]
    ax.table(cellText=rows,colLabels=['Model','ROC-AUC ± SD','Average precision ± SD'],cellLoc='left',bbox=[0,0,1,1])
    fig.text(.07,.2,'Sample SD over folds 3–4 and seeds 11, 23, 42. Each seed scores the same flights in its fold.',fontsize=10)
    pdf.savefig(fig);plt.close(fig)
    c=load('calibration')
    fig=page('B   How well do probabilities match outcomes?','Reliability curves for all eleven models  /  Each point represents at least 100 flights, seed 11.','On fold 3 XGBoost assigned 19,349 flights in its 10–20% bin a mean risk of 14.6%. The observed rate was 16.3%. On fold 4 the equivalent figures were 14.5% and 14.0%. Sparse high-risk bins are more uncertain. The full calibration table includes their counts.')
    axes=fig.subplots(1,2);fig.subplots_adjust(left=.08,right=.72,bottom=.27,top=.76,wspace=.29)
    for ax,(fold,g) in zip(axes,c.groupby('fold')):
        ax.plot([0,1],[0,1],'--',color='gray',lw=1)
        for i,(model,v) in enumerate(g.groupby('model')):
            v=v[v.flights>=100];ax.plot(v.predicted,v.observed,'.-',label=model,lw=1,color=plt.cm.tab20(i/19))
        ax.set(xlim=(0,1),ylim=(0,1),title=f'Fold {fold}',xlabel='Mean predicted risk',ylabel='Observed late fraction')
    axes[-1].legend(loc='center left',bbox_to_anchor=(1.03,.5),frameon=False)
    pdf.savefig(fig);plt.close(fig)
    def heat(frame,title,sub,note,cmap):
        fig=page(title,sub,note);ax=fig.add_axes([.24,.22,.65,.57])
        im=ax.imshow(frame,aspect='auto',cmap=cmap)
        ax.set_xticks(range(len(frame.columns)),frame.columns);ax.set_yticks(range(len(frame)),frame.index)
        for y,row in enumerate(frame.values):
            for x,val in enumerate(row):
                ax.text(x,y,f'{val:.2f}',va='center',ha='center',fontsize=9,bbox=dict(facecolor='white',alpha=.72,edgecolor='none',pad=1))
        fig.colorbar(im,ax=ax);pdf.savefig(fig);plt.close(fig)
    r=load('robustness').pivot(index='model',columns=['airport','fold'],values='ap')
    delta=r.xs(4,axis=1,level=1)-r.xs(3,axis=1,level=1)
    heat(delta,'D   Does performance carry across periods?','Change in AP from fold 3 to fold 4 within each origin  /  Positive values indicate improvement.','XGBoost improved in ATL (0.315 to 0.405) and ORD (0.402 to 0.478), but declined in LAX (0.402 to 0.341). ATL prevalence stayed near 18%. Airport-specific changes in ranking matter even when prevalence is stable. These observational comparisons do not identify a cause.','RdBu')
    e=load('errors');g=e[(e.fold==4)&(e.hours!='All')].pivot(index='model',columns='hours',values='recall')
    heat(g,'E   Which late departures escape an alert?','Fraction of late flights caught by local hour, fold 4  /  Each model flags the highest-risk 10% overall.','XGBoost caught 41.2% of late evening flights and 5.4% of late morning flights. Its 5,890 alerts included 2,823 false alarms and missed 8,516 delays. A common review capacity favours high-scoring periods. The tables include both folds, false alarms, misses and precision. Ties receive proportional credit.','BuGn')
