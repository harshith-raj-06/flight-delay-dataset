# Departure studies

Run from the repository root with the project environment and the existing result artifacts:

    .venv/bin/python site/build_site_data.py
    .venv/bin/python site/serve.py

Open http://127.0.0.1:8766/. If the local macOS service is already running, open that URL directly. The builder packages saved CSVs and the column audit into site/data. The test page uses the aggregate december_final.csv only. Deleting that summary and rebuilding produces the pending state.

The explorer requires final_csv/validation_T2h.csv and results/xgboost/model.joblib, plus the matching XGBoost and sklearn environment. Its airport/hour comparison uses the development validation export. The endpoint receives a development record ID and passes only the 35 approved features to the fitted pipeline. Labels are used for outcome reveal and the historical context statistic.

Run `.venv/bin/python site/feature_studies.py` from the repository root to reproduce the saved feature ladder, checkpoint comparison and XGBoost explanations in site/data/studies.json. This uses development folds 3 and 4 and the locked seed-11 model configuration. Explanations use a 1,500-flight fold-4 sample. Page views read the saved results. Desktop and mobile previews are in screenshots/.

The chart builder reads only the train and validation T−2h exports for the monthly rates and hour/weekday heatmap. ROC and precision–recall curves use saved validation probabilities at seed 11, grouped by reporting fold, and display up to 130 threshold points. Calibration uses saved bins with at least 100 flights. Each interactive graph has supporting data and a CSV download.
