"""Local multipage website. Run with the repository's project environment."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
import sys,json
import joblib
import pandas as pd

SITE=Path(__file__).resolve().parent
ROOT=SITE.parent
sys.path.append(str(ROOT))
from preprocessing import FEATURES
ROUTES={'/','/protocol','/models','/features','/interpretation','/errors','/explorer','/test','/method'}
data=pd.read_csv(ROOT/'final_csv/validation_T2h.csv',usecols=['row_id','FlightDate','LATE']+FEATURES).set_index('row_id')
pipe=joblib.load(ROOT/'results/xgboost/model.joblib')


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url=urlsplit(self.path)
        q={k:v[0] for k,v in parse_qs(url.query).items()}
        if url.path=='/api/flights':
            group=data[(data.Origin==q.get('airport','ATL'))&(data.FlightDate==q.get('date','2025-10-13'))]
            result=[{'id':int(i),'label':f'{r.dep_hour:02d}h · {r.Reporting_Airline} → {r.Dest} · #{i}'} for i,r in group.iterrows()]
            return self.send(json.dumps(result).encode(),'application/json')
        if url.path=='/api/predict':
            try:
                i=int(q.get('id','')); row=data.loc[[i]]; r=row.iloc[0]
            except (KeyError, ValueError):
                return self.send_error(404,'Unknown development record')
            result={'route':f'{r.Origin} → {r.Dest}','id':i,'date':r.FlightDate,'p':float(pipe.predict_proba(row[FEATURES])[0,1]),'actual':int(r.LATE),'context':float(data.loc[(data.Origin==r.Origin)&(data.dep_hour==r.dep_hour),'LATE'].mean()),'history':r.prior_mean_delay_60min_T2h,'age':r.wx_obs_age_min_T2h}
            return self.send(pd.Series(result).to_json().encode(),'application/json')
        if url.path in ROUTES:
            path=SITE/'index.html'
        elif url.path in ['/app.js','/style.css'] or url.path in ['/data/'+p.name for p in (SITE/'data').glob('*.json')]:
            path=SITE/url.path[1:]
        else:
            return self.send_error(404)
        self.send(path.read_bytes(),{'.html':'text/html','.js':'application/javascript','.css':'text/css','.json':'application/json'}[path.suffix]+'; charset=utf-8')

    def send(self,payload,kind):
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)


if __name__=='__main__':
    ThreadingHTTPServer(('127.0.0.1',8766),Handler).serve_forever()
