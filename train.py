from pathlib import Path
import json, joblib, sklearn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
ROOT=Path(__file__).resolve().parent
FEATURES=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
def train():
    df=pd.read_csv(ROOT/'data/Iris.csv')
    df=df.dropna(subset=FEATURES+['Species'])
    X=df[FEATURES]; y=df.Species
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    candidates={'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)), 'Decision Tree':DecisionTreeClassifier(random_state=42,max_depth=4), 'Random Forest':RandomForestClassifier(n_estimators=100,random_state=42)}
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    scores={name:float(cross_val_score(model,Xtr,ytr,cv=cv).mean()) for name,model in candidates.items()}
    name=max(scores,key=scores.get); model=candidates[name].fit(Xtr,ytr); pred=model.predict(Xte)
    metrics={'selected_model':name,'cv_accuracy':scores,'test_accuracy':float(accuracy_score(yte,pred)),'train_rows':len(Xtr),'test_rows':len(Xte),'classification_report':classification_report(yte,pred,output_dict=True,zero_division=0),'sklearn_version':sklearn.__version__}
    joblib.dump(model,ROOT/'models/iris_model.joblib')
    (ROOT/'reports/metrics.json').write_text(json.dumps(metrics,indent=2))
    ConfusionMatrixDisplay.from_predictions(yte,pred,xticks_rotation=20,cmap='Blues'); plt.tight_layout(); plt.savefig(ROOT/'reports/confusion_matrix.png',dpi=160); plt.close()
    for species,group in df.groupby('Species'):
        plt.scatter(group.PetalLengthCm,group.PetalWidthCm,label=species,alpha=.75)
    plt.xlabel('Petal length (cm)'); plt.ylabel('Petal width (cm)'); plt.legend(); plt.tight_layout(); plt.savefig(ROOT/'reports/eda.png',dpi=160); plt.close()
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': train()
