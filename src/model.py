"""
Model pipeline for Development Permit Approval Predictor.

Combines NLP features (TF-IDF on permit descriptions) with categorical
and numerical features to predict whether a development permit will be
approved.  Supports Logistic Regression, Random Forest, Gradient Boosting,
and XGBoost classifiers.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# Columns to label-encode
CATEGORICAL_COLS = [
    "category",
    "landusedistrict",
    "communityname",
    "quadrant",
    "permitteddiscretionary",
]

# Numerical columns that go straight into the feature matrix
NUMERICAL_COLS = [
    "applied_year",
    "applied_month",
    "applied_day_of_week",
]


class FeatureBuilder:
    """Build a combined feature matrix from preprocessed permit data.

    The builder creates:
    * TF-IDF features from the cleaned description text.
    * Label-encoded categorical features.
    * Pass-through numerical features.
    """

    def __init__(self, tfidf_max_features: int = 500):
        self.tfidf_max_features = tfidf_max_features
        self.tfidf: Optional[TfidfVectorizer] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._is_fitted = False

    # ---- fit / transform ------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        """Fit TF-IDF vectoriser and label encoders on *df*."""
        # TF-IDF
        self.tfidf = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
        )
        corpus = df["description_clean"].fillna("").astype(str)
        self.tfidf.fit(corpus)

        # Label encoders
        self.label_encoders = {}
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str).fillna("Unknown"))
                self.label_encoders[col] = le

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform *df* into a combined feature matrix (dense array)."""
        if not self._is_fitted:
            raise RuntimeError("FeatureBuilder has not been fitted yet.")

        # TF-IDF  (sparse)
        corpus = df["description_clean"].fillna("").astype(str)
        tfidf_matrix = self.tfidf.transform(corpus)

        # Categorical (dense)
        cat_parts: List[np.ndarray] = []
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                vals = df[col].astype(str).fillna("Unknown")
                # Handle unseen labels gracefully
                encoded = np.array(
                    [
                        le.transform([v])[0] if v in le.classes_ else -1
                        for v in vals
                    ]
                ).reshape(-1, 1)
                cat_parts.append(encoded)

        # Numerical (dense)
        num_parts: List[np.ndarray] = []
        for col in NUMERICAL_COLS:
            if col in df.columns:
                num_parts.append(
                    pd.to_numeric(df[col], errors="coerce")
                    .fillna(0)
                    .values.reshape(-1, 1)
                )

        # Combine
        dense_parts = cat_parts + num_parts
        if dense_parts:
            dense_matrix = np.hstack(dense_parts)
        else:
            dense_matrix = np.empty((len(df), 0))

        if issparse(tfidf_matrix):
            combined = hstack([tfidf_matrix, dense_matrix]).toarray()
        else:
            combined = np.hstack([tfidf_matrix, dense_matrix])

        return combined

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience: fit then transform."""
        return self.fit(df).transform(df)

    # ---- feature names --------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names matching column indices."""
        names: List[str] = []
        if self.tfidf is not None:
            names.extend(
                [f"tfidf__{t}" for t in self.tfidf.get_feature_names_out()]
            )
        for col in CATEGORICAL_COLS:
            if col in self.label_encoders:
                names.append(col)
        for col in NUMERICAL_COLS:
            names.append(col)
        return names

    def get_tfidf_feature_names(self) -> List[str]:
        """Return just the TF-IDF term names (without 'tfidf__' prefix)."""
        if self.tfidf is not None:
            return list(self.tfidf.get_feature_names_out())
        return []


# ---------------------------------------------------------------------------
# Classifier registry
# ---------------------------------------------------------------------------

CLASSIFIERS: Dict[str, Any] = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, solver="saga", random_state=42, n_jobs=-1
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    ),
    "XGBClassifier": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    ),
}


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Compute standard binary-classification metrics.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, auc_roc,
              confusion_matrix, fpr, tpr, roc_thresholds,
              classification_report (str).
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": thresholds,
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0
        ),
    }


def train_and_evaluate(
    df: pd.DataFrame,
    model_names: Optional[List[str]] = None,
    tfidf_max_features: int = 500,
    test_size: float = 0.2,
) -> Dict[str, Dict[str, Any]]:
    """End-to-end training pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe (must contain 'approved' and 'description_clean').
    model_names : list[str] or None
        Subset of CLASSIFIERS keys to train.  None trains all.
    tfidf_max_features : int
        Vocabulary size for TF-IDF vectoriser.
    test_size : float
        Fraction for test split.

    Returns
    -------
    dict
        ``{model_name: {"model": fitted_model,
                        "metrics": {...},
                        "feature_builder": FeatureBuilder}}``
    """
    if "approved" not in df.columns:
        raise ValueError("DataFrame must contain an 'approved' column.")

    if model_names is None:
        model_names = list(CLASSIFIERS.keys())

    # Build features
    fb = FeatureBuilder(tfidf_max_features=tfidf_max_features)
    X = fb.fit_transform(df)
    y = df["approved"].values.astype(int)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    logger.info(
        "Train size: %d | Test size: %d | Features: %d",
        X_train.shape[0],
        X_test.shape[0],
        X_train.shape[1],
    )

    results: Dict[str, Dict[str, Any]] = {}

    for name in model_names:
        if name not in CLASSIFIERS:
            logger.warning("Unknown model '%s'; skipping.", name)
            continue

        logger.info("Training %s ...", name)
        clf = CLASSIFIERS[name]
        clf.fit(X_train, y_train)
        metrics = evaluate_model(clf, X_test, y_test)
        logger.info(
            "%s  -->  Accuracy=%.4f  F1=%.4f  AUC=%.4f",
            name,
            metrics["accuracy"],
            metrics["f1"],
            metrics["auc_roc"],
        )
        results[name] = {
            "model": clf,
            "metrics": metrics,
            "feature_builder": fb,
        }

    return results


# ---------------------------------------------------------------------------
# Feature importance helpers
# ---------------------------------------------------------------------------

def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract feature importances from a fitted model.

    Works with tree-based models (``feature_importances_``) and linear
    models (``coef_``).

    Returns a DataFrame sorted by absolute importance descending.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_.ravel()
    else:
        logger.warning("Model has no recognized importance attribute.")
        return pd.DataFrame(columns=["feature", "importance"])

    imp_df = pd.DataFrame(
        {"feature": feature_names[: len(importances)], "importance": importances}
    )
    imp_df["abs_importance"] = imp_df["importance"].abs()
    imp_df.sort_values("abs_importance", ascending=False, inplace=True)
    return imp_df.head(top_n).reset_index(drop=True)


def get_tfidf_importance(
    model: Any,
    feature_builder: FeatureBuilder,
    top_n: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return top TF-IDF terms driving approval and refusal.

    Returns
    -------
    (approval_df, refusal_df)
        Each has columns [term, weight].
    """
    tfidf_names = feature_builder.get_tfidf_feature_names()
    n_tfidf = len(tfidf_names)

    if hasattr(model, "coef_"):
        weights = model.coef_.ravel()[:n_tfidf]
    elif hasattr(model, "feature_importances_"):
        weights = model.feature_importances_[:n_tfidf]
    else:
        return pd.DataFrame(columns=["term", "weight"]), pd.DataFrame(
            columns=["term", "weight"]
        )

    term_df = pd.DataFrame({"term": tfidf_names, "weight": weights})

    # For linear models positive coef => approval, negative => refusal
    if hasattr(model, "coef_"):
        approval_df = (
            term_df.nlargest(top_n, "weight")[["term", "weight"]].reset_index(
                drop=True
            )
        )
        refusal_df = (
            term_df.nsmallest(top_n, "weight")[["term", "weight"]].reset_index(
                drop=True
            )
        )
        refusal_df["weight"] = refusal_df["weight"].abs()
    else:
        # For tree-based, split by above/below median
        sorted_df = term_df.nlargest(top_n * 2, "weight").reset_index(drop=True)
        approval_df = sorted_df.head(top_n).reset_index(drop=True)
        refusal_df = sorted_df.tail(top_n).reset_index(drop=True)

    return approval_df, refusal_df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_artifacts(
    model: Any,
    feature_builder: FeatureBuilder,
    model_name: str = "best_model",
) -> Path:
    """Save model and feature builder to the models/ directory."""
    artifact = {
        "model": model,
        "feature_builder": feature_builder,
    }
    path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(artifact, path)
    logger.info("Saved model artifacts to %s", path)
    return path


def load_artifacts(model_name: str = "best_model") -> Dict[str, Any]:
    """Load model and feature builder from the models/ directory."""
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    artifact = joblib.load(path)
    logger.info("Loaded model artifacts from %s", path)
    return artifact


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_and_preprocess

    df = load_and_preprocess()
    results = train_and_evaluate(df)

    # Save best model by AUC
    best_name = max(results, key=lambda k: results[k]["metrics"]["auc_roc"])
    best = results[best_name]
    print(f"\nBest model: {best_name}")
    print(best["metrics"]["classification_report"])
    save_artifacts(best["model"], best["feature_builder"], model_name="best_model")
