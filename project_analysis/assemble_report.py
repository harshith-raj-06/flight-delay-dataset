"""Assemble the project PDF from previously computed results. Requires reportlab and pypdf."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from pypdf import PdfWriter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

here=Path(__file__).resolve().parent
repo=here.parent
styles=getSampleStyleSheet()
story=[]
def para(text,style='BodyText'):
    story.extend([Paragraph(text,styles[style]),Spacer(1,8)])

def table(rows,widths):
    data=[[Paragraph(str(value),styles['BodyText']) for value in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#cce2dd')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f6f5')]),('VALIGN',(0,0),(-1,-1),'TOP')]))
    story.extend([t,Spacer(1,10)])

para('Model setup and tuning','Title')
para('Folds 1 and 2 select hyperparameters by average precision at seed 11. The later reporting folds use locked settings. The table lists the selected settings stored alongside each fitted pipeline.')
rows=[['Model','Selected settings']]
for f in sorted((repo/'results').glob('*/settings.json')):
    settings=json.loads(f.read_text())
    config=settings.get('best_parameters') or settings.get('best_params') or settings.get('params')
    rows.append([f.parent.name.replace('_',' '),', '.join(f'{k} = {v}' for k,v in config.items())])
table(rows,[140,345])
para('The decision tree selected depth 16 and at least 500 observations per leaf. Validation favoured zero additional cost-complexity pruning under this leaf constraint. The final tree had 836 leaves. The tuning files preserve the alternatives and scores.')
para('Acquisition and scope','Heading2')
para('BTS and IEM downloads were logged on November 25, 2026. BTS distributes carrier-reported monthly records through TranStats. A separate licence for the monthly archive was not verified. IEM publishes its archive usage policy. Source links: <link href="https://transtats.bts.gov/">BTS</link> and <link href="https://mesonet.agron.iastate.edu/disclaimer.php">IEM terms</link>.')
story.append(PageBreak())
para('Dimensionality reduction','Title')
para('PCA targets 95% training variance. It compressed 76 encoded features to 30 components in tuning folds and 31 in the reporting folds. Relative to plain logistic regression, the PCA pipeline lost 0.0174 ROC-AUC and 0.0148 average precision. Variance retention measures reconstruction, while delay prediction depends on information associated with the outcome.')
im=Image(str(repo/'results/pca_logistic_regression/pca_variance.png'))
im.drawHeight=im.drawHeight*470/im.drawWidth;im.drawWidth=470;story.append(im)
para('Scope of inference','Heading2')
para('The study covers three busy US origins and one calendar year. Weather observation time is a proxy for information availability. Airline reporting delays, tail rotations and operational decisions may alter real-time performance. Earlier prototypes inspected parts of the development timeline. A previous file search incidentally surfaced December records without computing test metrics. The final scoring run fixed all pipelines before loading December for measurement.')
story.append(PageBreak())
para('Final December evaluation','Title')
scores=list(csv.DictReader((here/'december_final.csv').open()))
para(f'Test start: December 1, 2025. End: December 31. All eleven previously fitted pipelines scored the same 68,146 flights. The observed late rate was {float(scores[0]["late_rate"]):.2%}. Only the 35 checkpoint features entered inference. The stored freeze manifest identifies the pipelines by SHA-256 digest.')
table([['Model','Test ROC-AUC','Test average precision']]+[[r['model'].replace('_',' '),f"{float(r['roc_auc']):.4f}",f"{float(r['average_precision']):.4f}"] for r in sorted(scores,key=lambda v:float(v['average_precision']),reverse=True)],[210,125,155])
para('XGBoost was the development leader by mean average precision and supplies the demonstration UI. The test covers one month, so test statistics are single-period estimates. No model was selected or tuned using these scores.')
para('Reproduction and interactive demonstration','Heading2')
para('The local flight explorer runs the fitted XGBoost predictor for selected development flights. It displays risk alongside the previous hour’s departures, mean delay, visibility and weather observation age. The project_analysis directory contains analysis code, data tables, figure generation and launch instructions. <link href="https://github.com/harshith-raj-06/flight-delay-dataset">GitHub repository</link>.')
with TemporaryDirectory() as temp:
    appendix=Path(temp)/'supplement.pdf'
    SimpleDocTemplate(str(appendix),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=36,bottomMargin=36).build(story)
    w=PdfWriter()
    for path in [repo.parent/'output/pdf/Flight_Dataset_and_Preprocessing_Report.pdf',here/'Analysis_and_benchmark.pdf',appendix]:
        w.append(path)
    w.add_metadata({'/Title':'Flight departure delay at T−2h | Project report'})
    w.write(here/'Flight_Delay_Project_Report.pdf')
