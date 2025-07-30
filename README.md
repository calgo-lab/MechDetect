# MechDetect
Code accompanying the MechDetect paper

## Description
- `src/auc_roc_scores` contains the code to obtain the AUC-ROC scores that are used in MechDetect.
    - `src/auc_roc_scores/testing.py` is the code that trains models to predict error masks and records the AUC-ROC values.
    - `src/auc_roc_scores/my-job-chart/run_all_experiments.sh` contains all OpenML dataset ids for the datasets on which the experiments were run.
- `src/auc_roc_analysis.ipynb` contains the code associated with the analysis of the AUC-ROC scores under various mechanisms and tasks.
- `src/accuracy_analysis.ipynb` contains the code associated with the analysis of the accuracy of MechDetect.
- `plots` contains the plots produced by the code for the paper.
- `data` contains the tensors of AUC-ROC scores for the various instances of the experiments.
