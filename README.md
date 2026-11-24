# Iris Flower Species Classification

## Run on Windows (Python 3.12)
Extract the ZIP and double-click START_PROJECT.bat. Internet is needed for the first package installation. The script creates an isolated environment, installs dependencies, trains models and opens http://127.0.0.1:8501. Keep the command window open; Ctrl+C stops the server.

Alternatively run in PowerShell from this folder:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe app.py
```

## Dataset provenance
Selected Kaggle dataset: https://www.kaggle.com/datasets/uciml/iris (CC0). Direct Kaggle download was unavailable in the build environment. The bundled CSV is generated from scikit-learn's corrected classic Iris dataset, with Kaggle-compatible column names. It is not a verified byte-for-byte Kaggle download. To use Kaggle's exact version, download Iris.csv from that link, replace data/Iris.csv, and rerun train.py. Original dataset: https://archive.ics.uci.edu/dataset/53/iris. Loader: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html.

## Workflow
1. Load 150 labeled flower measurements; check missing values; exclude identifier from features.
2. Explore petal lengths and widths using the scatter plot.
3. Split into 120 training and 30 held-out test rows, stratified by species.
4. Compare logistic regression (with scaling inside a pipeline), decision tree and random forest via five-fold cross-validation on training data only.
5. Select the highest mean CV score, fit on training data, and evaluate once on held-out data. Save model, classification report and confusion matrix.
6. Enter four measurements in the browser UI to receive a species prediction and estimated probabilities.

## Results on bundled data
Logistic regression selected (95.83% mean CV accuracy); held-out accuracy 93.33% (28/30). Small dataset, so results vary with split and do not establish real-world performance. See reports/metrics.json for precision, recall and F1 by class. Input outside the training distribution may produce unreliable predictions.

## Interview explanation
The goal is to predict flower type based on numerical measurements. This is supervised multiclass classification because the training examples already have labels and there are three possible classes. The training set teaches the model; validation compares algorithms; the untouched test set measures generalization. A confusion matrix shows correct and incorrect predictions for each class. The saved pipeline applies the same scaling at prediction time as training.

## Example inputs (cm)
- 5.1, 3.5, 1.4, 0.2: Setosa
- 6.7, 3.0, 5.2, 2.3: Virginica

## Files
- train.py: training, evaluation, charts
- app.py: local HTTP application and prediction logic
- data/Iris.csv: offline dataset
- models/iris_model.joblib: fitted model (load only trusted artifacts; retrain after dependency changes)
- reports/: actual evaluation outputs
- START_PROJECT.bat: Windows launcher
- requirements.txt: dependencies
