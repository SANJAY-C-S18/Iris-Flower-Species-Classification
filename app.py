from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs
from html import escape
import json, math, joblib, pandas as pd
from train import FEATURES
ROOT=Path(__file__).resolve().parent
MODEL=joblib.load(ROOT/'models/iris_model.joblib')
METRICS=json.loads((ROOT/'reports/metrics.json').read_text())
LABELS=['Sepal length','Sepal width','Petal length','Petal width']
def predict(values):
    if len(values)!=4 or any(not math.isfinite(v) or not 0<v<=20 for v in values): raise ValueError('Enter four positive measurements up to 20 cm.')
    frame=pd.DataFrame([values],columns=FEATURES)
    return str(MODEL.predict(frame)[0]),dict(zip(MODEL.classes_,MODEL.predict_proba(frame)[0]))
def page(values=None,result=''):
    values=values or [5.1,3.5,1.4,.2]
    fields=''.join(f'<label>{label} (cm)<input name="{key}" type="number" min="0.01" max="20" step="0.01" required value="{escape(str(v),quote=True)}"></label>' for label,key,v in zip(LABELS,FEATURES,values))
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Iris Classifier</title><style>body{{font:16px system-ui;background:#eef3ef;color:#17382d;margin:0}}main{{max-width:850px;margin:40px auto;padding:24px}}section{{background:white;padding:24px;border-radius:16px;margin:18px 0}}form{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}}label{{display:grid;gap:8px}}input,button{{padding:12px;font:inherit;border:1px solid #aabbb3;border-radius:8px}}button{{background:#176b50;color:white;cursor:pointer}}img{{max-width:100%}}small{{color:#53685e}}</style><main><small>SUPERVISED MACHINE LEARNING / MULTICLASS</small><h1>Iris Species Classifier</h1><p>Predict a flower species from four measurements.</p><section><b>{escape(METRICS['selected_model'])}</b> · Held-out accuracy: {METRICS['test_accuracy']:.1%}<form method="post" action="/predict">{fields}<button>Predict species</button></form>{result}</section><section><h2>Model evaluation</h2><p>120 training samples · 30 test samples</p><img src="/confusion_matrix.png" alt="Confusion matrix"><h2>Dataset exploration</h2><img src="/eda.png" alt="Petal measurements by species"></section><small>Educational model. Scores are model estimates, not guarantees.</small></main></html>'''
class Handler(BaseHTTPRequestHandler):
    def send(self,body,status=200,kind='text/html; charset=utf-8'):
        if isinstance(body,str): body=body.encode()
        self.send_response(status); self.send_header('Content-Type',kind); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path=='/': self.send(page())
        elif self.path in ['/eda.png','/confusion_matrix.png']: self.send((ROOT/'reports'/self.path[1:]).read_bytes(),kind='image/png')
        else: self.send('Not found',404)
    def do_POST(self):
        if self.path!='/predict': return self.send('Not found',404)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=4096: raise ValueError('Invalid request size')
            data=parse_qs(self.rfile.read(n).decode()); values=[float(data[k][0]) for k in FEATURES]; name,probs=predict(values)
            result='<h2>'+escape(name)+'</h2>'+''.join(f'<p>{escape(str(k))}: {v:.1%}</p>' for k,v in probs.items()); self.send(page(values,result))
        except (ValueError,KeyError,IndexError): self.send(page(result='<p>Invalid measurements. Please enter four positive values up to 20 cm.</p>'),400)
if __name__=='__main__':
    import webbrowser, threading
    server=ThreadingHTTPServer(('127.0.0.1',8501),Handler)
    print('Open http://127.0.0.1:8501 (Ctrl+C to stop)')
    threading.Timer(.5,lambda:webbrowser.open('http://127.0.0.1:8501')).start()
    server.serve_forever()
