# Flight departure analysis

Run from the repository root in the project environment.

    python project_analysis/analyze.py
    python project_analysis/report.py


## Analysis design

The marking scheme requests three systematic analyses. B compares probability bins with observed outcomes. D measures sensitivity to temporal distribution changes within each airport using AP, AP divided by local prevalence, and ROC-AUC. E compares precision, recall, false alerts and misses at a fixed review capacity by local departure hour. Tables cover all eleven models. Slice analyses use the prespecified common seed 11. The main benchmark recomputes all six fold/seed scores. No December data is needed for these commands.

## Viva prompts

Explain why a prior-hour departure outcome is available at T, how report age is bounded, why a majority probability has AP equal to prevalence, why PCA can discard predictive signal, and why improving overall AP can coexist with deteriorating performance at one airport. Discuss the capacity trade-off using false alerts and missed delays.

## Final scoring record

December scoring has been completed. Read december_final.csv and december_freeze.json. The scorer preflighted the trusted fitted models on development rows, isolated the four scikit-learn 1.6.1 models, locked their hashes and passed only the 35 inputs to inference. The other seven models ran with scikit-learn 1.9.1. The exclusive freeze file prevents accidental repeat evaluation. Use the saved CSV to rebuild the report.

The optional PDF assembly script uses reportlab and pypdf and reads the existing dataset report from ../output/pdf relative to the repository. Benchmark and analysis PDF generation uses matplotlib.
