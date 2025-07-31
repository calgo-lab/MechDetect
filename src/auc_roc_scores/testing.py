import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.ensemble import HistGradientBoostingClassifier
import sys
import os
from sklearn.model_selection import cross_val_score
from tab_err.api import high_level
from tab_err import error_type, error_mechanism
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OrdinalEncoder



# Outline:

# We are given one dataset and error rate
# Want to do the following for each error mechanism with a fixed (missing value error type)

# Read in cl args, same exact ones as error_mechanisms experiments
# Read in the dataset
# Use tab err to get errored data and mask
# For df in [dirty data, clean data]:
#   For each column in the dataset:
#       Train model function case 1
#       Train model function case 2
#       Train model function case 3
#       Train model function case 4
#       Append results to numpy array
# Write metadata file (how to read numpy array)
# Write the numpy array

def train_and_evaluate_models(data: pd.DataFrame, target: pd.DataFrame, column: str | int, model: BaseEstimator, seed: int, cv_folds: int = 10, n_jobs: int = 1):
    """Splits data in the given ways and trains/ evaluates a model"""
    # Break up data into cases
    score = make_scorer(roc_auc_score)
    
    # We did this in the other experiments too to add resiliency for string/object typed items
    categorical_columns = data.select_dtypes(include=["object"]).columns
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    data[categorical_columns] = encoder.fit_transform(data[categorical_columns].astype(str))
    
    # Case 1 (all data, error mask):
    case_1_scores = cross_val_score(model, data, y=target, cv=cv_folds, scoring=score, n_jobs=n_jobs)
    
    # Case 2 (all data, scrambled mask):
    case_2_target = target.sample(frac=1, random_state=seed)
    case_2_scores = cross_val_score(model, data, y=case_2_target, cv=cv_folds, scoring=score, n_jobs=n_jobs)
    
    # Case 3 (w/o column, error mask):
    case_3_data = data.drop(columns=[column])
    case_3_scores = cross_val_score(model, case_3_data, y=target, cv=cv_folds, scoring=score, n_jobs=n_jobs)
    
    # Case 4 (only column, error mask):
    case_4_data = pd.DataFrame(data[column])
    case_4_scores = cross_val_score(model, case_4_data, y=target, cv=cv_folds, scoring=score, n_jobs=n_jobs)
    
    resulting_array = np.array([case_1_scores, case_2_scores, case_3_scores, case_4_scores]) # index 0 is case, index 1 is cv score
    
    return resulting_array
    


def main():
    # Read in arguments
    if len(sys.argv) != 6:
        print("Usage: python to_cluster_test.py <n-workers> <random_seed> <experiment-name> <dataset_id> <error_rate>")
        sys.exit(1)

    # Experiements Setup
    n_jobs = int(sys.argv[1])  # Testing, set in the helm job.yml
    seed = int(sys.argv[2])
    experiment_name = sys.argv[3]
    dataset_id = sys.argv[4]
    error_rate = float(sys.argv[5])
    folds = 10
    
    # Model setup 
    model = HistGradientBoostingClassifier(max_iter=200)
    
    # Set up filesystem paths
    dataset_directory = "./datasets"
    results_directory = f"./results/{experiment_name}/{dataset_id}/{error_rate}"
    
    os.makedirs(results_directory, exist_ok=True)
    dataset_path = os.path.join(dataset_directory, f"{dataset_id}.csv")
    result_array_path = os.path.join(results_directory, "results.npy")  # As binarized numpy file
    results_metadata_path = os.path.join(results_directory, "metadata.json")
    results_finished_path = os.path.join(results_directory, "FINISHED")
    
    data = pd.read_csv(dataset_path)
    data = data.drop(columns=["target"])
    
    
    error_mechs_array = []
    error_mech_order = ["ECAR", "EAR", "ENAR"]
    for mech in error_mech_order:
        columns_array = []
        
        # Different error patterns
        if mech == "ECAR":
            dirty_data, error_mask = high_level.create_errors(data, error_rate=error_rate, error_mechanisms_to_include=[error_mechanism.ECAR()], error_types_to_include=[error_type.MissingValue()], seed=seed)
        elif mech == "EAR":  # Note it is currently done randomly, we could set it to use the first column (or the target to condition all others)
            dirty_data, error_mask = high_level.create_errors(data, error_rate=error_rate, error_mechanisms_to_exclude=[error_mechanism.ECAR(), error_mechanism.ENAR()], error_types_to_include=[error_type.MissingValue()], seed=seed)
        else:  # ENAR (if the error_mech_order array is correct)
            dirty_data, error_mask = high_level.create_errors(data, error_rate=error_rate, error_mechanisms_to_include=[error_mechanism.ENAR()], error_types_to_include=[error_type.MissingValue()], seed=seed)
        
        
        for col in data.columns:
            clean_dirty_array = []
            
            for clean in [True, False]:
                error_mask_column = error_mask[col]
                if clean:
                    cv_results_array = train_and_evaluate_models(data=data, target=error_mask_column, column=col, model=model, seed=seed, cv_folds=folds, n_jobs=n_jobs)  # Should be (case, cv_iter)
                else:
                    cv_results_array = train_and_evaluate_models(data=dirty_data, target=error_mask_column, column=col,  model=model, seed=seed, cv_folds=folds, n_jobs=n_jobs)    
                print("cv_results array shape: ", cv_results_array.shape)
                # Accumulation step
                clean_dirty_array.append(cv_results_array)
            
            # Accumulation step
            stacked_clean_dirty_array = np.stack(clean_dirty_array, axis=0)  # 3D - (c/d, case, cv_iter)
            columns_array.append(stacked_clean_dirty_array)
            print("Clean/Dirty shape: ", stacked_clean_dirty_array.shape)
        
        # Accumulation step
        stacked_columns_array = np.stack(columns_array, axis=0)  # 4D - (column, c/d, case, cv_iter)
        error_mechs_array.append(stacked_columns_array)
        print("Columns array shape", stacked_columns_array.shape)
    
    full_results_array = np.stack(error_mechs_array, axis=0)  # 5D - (error_mech, column, c/d, case, cv_iter)
    print("Full results array shape: ", full_results_array.shape)
    
    # Write up experiment in metadata file
    experiment_metadata = {
        "CV Folds": folds,
        "Error Mechanism Order": error_mech_order,
        "Dimensions of NumPy Array": full_results_array.shape,
        "Explanation of Dimensions": "(error_mechanism, column, clean/dirty, case, cv_iter)",
        "Overview Notes": f"This experiment looked at dataset {dataset_id} and for each error mechanism, trained a model on the clean and dirty data of 4 different cases, returning cross validation results from each model training.",
        "Case Ordering": "Case 1: Train on all data to predict column error mask, Case 2: Train on all data to predict shuffled column error mask, Case 3: Train on all but 'column' data columns to predict column error mask, Case 4: Train a model only on column 'column' to predict column error mask."
    }

    # Write results
    np.save(result_array_path, full_results_array)
    
    with open(results_metadata_path, "w") as f:
        json.dump(experiment_metadata, f, indent=4)

    with open(results_finished_path, "w") as _:
        pass
    
if __name__ == "__main__":
    main()

