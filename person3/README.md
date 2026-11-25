# Person 3 - Gradient-Boosting Training Guide

This folder is your complete Person 3 submission. The notebook compares four gradient-boosting models for predicting whether a flight will depart 15 or more minutes late, two hours before scheduled departure.

## What is in this folder?

- `person3_gradient_boosting_T2h.ipynb` - the notebook you run to train the models.
- `README.md` - this step-by-step guide.

After training, the notebook creates a separate `person3_outputs/` folder containing the files Person 4 needs:

- `person3_summary_scores.csv` - final model comparison.
- `person3_fold_scores.csv` - score for each chronological validation fold.
- `person3_oof_predictions.parquet` - validation probability predictions.
- `person3_early_stopping.csv` - number of training iterations chosen for each run.
- `person3_chosen_settings.json` - seeds, features, folds, and early-stopping details.

## Before you start

The notebook needs the project dataset files in `final_csv/`:

- `train_T2h.csv`
- `validation_T2h.csv`

Do **not** train on `test_T2h.csv`. December is the final evaluation data and is reserved for Person 4.

If `final_csv/` is absent, it has not been built or downloaded yet. Ask the person responsible for data preparation for those two CSV files, or run the repository's data-build process from the project root:

```bash
python3 -m pip install -r requirements.txt
python3 pipeline/bts.py
python3 pipeline/iem.py
python3 build_features.py
```

That download/build can take a long time and needs a stable internet connection.

## Easiest option: Google Colab

1. Go to [Google Colab](https://colab.research.google.com/) and sign in with your Google account.
2. Choose **File -> Open notebook -> GitHub**.
3. Paste this repository URL: `https://github.com/harshith-raj-06/flight-delay-dataset`.
4. Open `person3/person3_gradient_boosting_T2h.ipynb`.
5. Keep the default **CPU** runtime for your first run. The notebook is configured for reliable CPU training; it may take several hours in full mode.
6. Upload the two required CSVs to the Colab session, keeping the folder name `final_csv`. The simplest way is to upload a zip named `final_csv.zip`, then run this in a new cell:

   ```python
   !unzip -q final_csv.zip
   ```

7. Run the installation cell at the top of the notebook. If Colab asks to restart the session, use **Runtime -> Restart session**, then rerun cells from the top.
8. In the configuration cell, keep `RUN_MODE = 'smoke'` for the first run. Run every cell in order. It checks that the data, dependencies, and code work.
9. Change `RUN_MODE = 'full'`, then choose **Runtime -> Run all**. Do not close the browser tab while it runs.
10. When the summary table appears, download the generated `person3_outputs/` folder. In the left-side Files panel, right-click each output file and select **Download**.

## Kaggle alternative

1. Create/sign in to a [Kaggle](https://www.kaggle.com/) account.
2. Create a new **Notebook** and enable **Internet** only if you need it for package installation.
3. Upload `person3_gradient_boosting_T2h.ipynb`, or upload the whole repository as a Kaggle Dataset and add it to the notebook.
4. Add a dataset containing `train_T2h.csv` and `validation_T2h.csv`.
5. Update `DATA_DIR` in the configuration cell to Kaggle's input path, for example:

   ```python
   DATA_DIR = Path('/kaggle/input/your-dataset-name/final_csv')
   OUT_DIR = Path('/kaggle/working/person3_outputs')
   ```

6. Run the smoke mode first, then change to full mode and run all cells.
7. Download files from the notebook's **Output** panel when training is complete.

## Which model should I report?

Use `person3_summary_scores.csv` after the full run. The best Person 3 model is normally the one with the largest `ap_mean` (Average Precision / PR-AUC). Do not use the December test file to decide. Send Person 4 all five output files, not just the best score.

## Expected training time

Full training runs 4 model families across 4 chronological folds and 3 random seeds (48 fits total). It may take hours on a CPU, so keep the Colab tab open and plugged in while it runs.
