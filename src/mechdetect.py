# Implementation of the MechDetect algorithm.
from sklearn.metrics import roc_auc_score, make_scorer
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import scipy.stats as stats

class MechDetector:
    """Detects the error-generation mechanism of a column using the MechDetect algorithm.

    The algorithm frames error-mechanism detection as a supervised learning task:
    a classifier is trained to predict whether an entry is erroneous (via a mask),
    and its performance is compared across three controlled scenarios using
    nonparametric hypothesis tests.

    Supported error mechanisms:
        - ECAR: Error Completely At Random
        - EAR:  Error At Random
        - ENAR: Error Not At Random

    Parameters
    ----------
    seed : int or None, optional
        Random seed used for reproducibility when shuffling the target.
    alpha : float, default=0.05
        Significance level for hypothesis testing. Bonferroni-corrected internally.
    n_jobs : int, default=1
        Number of threads used by CatBoost during training.
    cv_folds : int, default=10
        Number of cross-validation folds used to estimate model performance.

    Notes
    -----
    - Uses CatBoostClassifier to naturally support mixed numerical/categorical data.
    - Model performance is measured using ROC AUC.
    - Statistical testing is performed using the Mann–Whitney U test.
    """

    def __init__(self, seed: int | None = None, alpha: float = 0.05, n_jobs: int = 1, cv_folds: int = 10):
        """
        Initialize a MechDetector instance.
        """
        self.seed = seed
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds

        if alpha > 1.0 or alpha <= 0.0:
            raise ValueError(f"Alpha should be between 0 and 1. Got {alpha}")


    def _train_and_evaluate_models(
        self,
        data: pd.DataFrame,
        mask: pd.DataFrame,
        column: str | int,
    ) -> np.ndarray:
        """Train and evaluate predictive models under three experimental conditions.

        For a given column, this method estimates model performance (ROC AUC)
        using cross-validation for:

            1. Complete Task: Predict the true error mask using all columns.
            2. Shuffled Task: Predict a randomly permuted error mask using all columns.
            3. Missing Task:  Predict the true error mask with the target column removed.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset containing features.
        mask : pd.DataFrame
            Binary error mask aligned with `data`.
        column : str or int
            Column whose error mechanism is being evaluated.

        Returns
        -------
        np.ndarray
            Array of shape (3, cv_folds) containing ROC AUC scores for
            [complete, shuffled, missing] tasks.
        """
        score = make_scorer(roc_auc_score)
        target = mask[column]
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)  # Ensure that CV works!

        # Identify categorical features
        cat_cols = data.select_dtypes(include=["object", "category"]).columns
        cat_feature_indices = [data.columns.get_loc(c) for c in cat_cols]

        model = CatBoostClassifier(
            iterations=300,
            loss_function="Logloss",
            eval_metric="AUC",
            allow_writing_files=False,
            verbose=False,
            thread_count=self.n_jobs,
            cat_features=cat_feature_indices,
        )

        # Case 1: Complete task
        case_1_scores = cross_val_score(
            model, data, y=target, cv=skf, scoring=score
        )

        # Case 2: Shuffled target
        case_2_target = target.sample(frac=1, random_state=self.seed)
        case_2_scores = cross_val_score(
            model, data, y=case_2_target, cv=skf, scoring=score
        )

        # Case 3: Missing column
        case_3_data = data.drop(columns=[column])
        case_3_cat_cols = case_3_data.select_dtypes(include=["object", "category"]).columns
        case_3_cat_feature_indices = [
            case_3_data.columns.get_loc(c) for c in case_3_cat_cols
        ]

        case_3_model = CatBoostClassifier(
            iterations=300,
            loss_function="Logloss",
            eval_metric="AUC",
            allow_writing_files=False,
            verbose=False,
            thread_count=self.n_jobs,
            cat_features=case_3_cat_feature_indices,
        )

        case_3_scores = cross_val_score(
            case_3_model, case_3_data, y=target, cv=skf, scoring=score
        )

        return np.array([case_1_scores, case_2_scores, case_3_scores])


    def _decide_error_mech(
        self,
        case_1: np.ndarray,
        case_2: np.ndarray,
        case_3: np.ndarray,
    ) -> tuple[str, float, float | None]:
        """Infer the error mechanism via a two-stage statistical testing procedure.

        The decision process follows:

        Step 1:
            Test whether the complete task outperforms the shuffled task.
            If not, conclude ECAR.

        Step 2 (conditional):
            Test whether the complete task outperforms the missing-column task.
            If yes, conclude ENAR; otherwise, conclude EAR.

        All tests are one-sided Mann-Whitney U tests with Bonferroni correction.

        Parameters
        ----------
        case_1 : np.ndarray
            Cross-validated scores for the complete task.
        case_2 : np.ndarray
            Cross-validated scores for the shuffled task.
        case_3 : np.ndarray
            Cross-validated scores for the missing-column task.

        Returns
        -------
        tuple
            (mechanism, p_value_1, p_value_2)

            - mechanism : {"ECAR", "EAR", "ENAR"}
            - p_value_1 : float
                P-value for complete vs. shuffled test.
            - p_value_2 : float or None
                P-value for complete vs. missing test (None if not performed).
        """
        _, p_value_1 = stats.mannwhitneyu(
            case_1, case_2, alternative="greater"
        )
        _, p_value_2 = stats.mannwhitneyu(
            case_1, case_3, alternative="greater"
        )

        if p_value_1 < self.alpha / 2:
            if p_value_2 < self.alpha / 2:
                return ("ENAR", p_value_1, p_value_2)
            return ("EAR", p_value_1, p_value_2)

        return ("ECAR", p_value_1, None)


    def detect(
        self,
        data: pd.DataFrame,
        mask: pd.DataFrame,
        column: int | str,
    ) -> tuple[str, float, float | None]:
        """Run the full MechDetect pipeline for a single column.

        This method validates inputs, trains models under the three experimental
        conditions, and applies statistical testing to determine the most likely
        error mechanism.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset.
        mask : pd.DataFrame
            Binary error mask aligned with `data`.
        column : int or str
            Column index or name to analyze.

        Returns
        -------
        tuple
            (mechanism, p_value_1, p_value_2)

            See `_decide_error_mech` for interpretation.
        """
        if data.shape != mask.shape:
            raise ValueError(
                f"Mask and data must be the same shape. "
                f"Got data: {data.shape}, mask: {mask.shape}."
            )

        if data.shape[0] < self.cv_folds:
            raise ValueError(
                "Number of observations must be >= number of CV folds. "
                f"Got {data.shape[0]} observations and {self.cv_folds} folds."
            )

        if isinstance(column, str) and column not in data.columns:
            raise ValueError(f"Column {column} is not in data columns.")

        if isinstance(column, int) and column >= data.shape[1]:
            raise ValueError(f"Column index {column} is out of bounds.")

        case_1, case_2, case_3 = self._train_and_evaluate_models(data, mask, column)
        return self._decide_error_mech(case_1, case_2, case_3)
