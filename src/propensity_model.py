from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

# Model: prefer XGBoost if installed; else fallback to HistGradientBoosting
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import joblib
except Exception:
    joblib = None


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute KS statistic for binary classification."""
    df = pd.DataFrame({"y": y_true, "p": y_score}).sort_values("p")
    # cumulative distributions
    cdf_pos = (df["y"] == 1).cumsum() / max((df["y"] == 1).sum(), 1)
    cdf_neg = (df["y"] == 0).cumsum() / max((df["y"] == 0).sum(), 1)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def load_features() -> pd.DataFrame:
    # Prefer named clusters if present
    p1 = Path("data/processed/customer_features_with_cluster_named.csv")
    p2 = Path("data/processed/customer_features_with_cluster.csv")

    if p1.exists():
        return pd.read_csv(p1)
    if p2.exists():
        return pd.read_csv(p2)

    raise FileNotFoundError(
        "Could not find clustered features. Run clustering first:\n"
        "- python -m src.build_features\n"
        "- python -m src.cluster_profiles"
    )


def main() -> None:
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    df = load_features()

    # Target: adopted_ever (from build_features)
    if "adopted_ever" not in df.columns:
        raise ValueError("Target column 'adopted_ever' not found in features table.")

    y = df["adopted_ever"].astype(int).to_numpy()

    # Features: use m12_* and some basics + cluster (numeric)
    # (Keep it simple and stable for portfolio.)
    candidate_num = [
        "age",
        "income",
        "m12_mean_balance",
        "m12_std_balance",
        "m12_mean_card_spend",
        "m12_mean_utilization",
        "m12_mean_pix",
        "m12_late_payment_rate",
        # optional short window
        "m3_mean_balance",
        "m3_std_balance",
        "m3_mean_card_spend",
        "m3_mean_utilization",
        "m3_mean_pix",
        "m3_late_payment_rate",
        # cluster id (numeric)
        "cluster",
    ]
    num_cols = [c for c in candidate_num if c in df.columns]

    # Optional categorical (cluster_name if exists)
    cat_cols = []
    if "cluster_name" in df.columns:
        cat_cols.append("cluster_name")

    # Build X
    keep_cols = ["customer_id"] + num_cols + cat_cols
    X = df[keep_cols].copy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Preprocess
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    # Model
    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
        )
    else:
        model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.06,
            random_state=42,
        )

    clf = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    # Fit
    clf.fit(X_train.drop(columns=["customer_id"]), y_train)

    # Predict
    p_test = clf.predict_proba(X_test.drop(columns=["customer_id"]))[:, 1]
    auc = roc_auc_score(y_test, p_test)
    ks = ks_statistic(y_test, p_test)

    # Save metrics
    metrics = pd.DataFrame([{"auc": auc, "ks": ks, "model": "XGBoost" if HAS_XGB else "HistGB"}])
    metrics_path = Path("reports/tables/propensity_metrics.csv")
    metrics.to_csv(metrics_path, index=False)

    # Score all customers
    p_all = clf.predict_proba(X.drop(columns=["customer_id"]))[:, 1]
    scores = pd.DataFrame({"customer_id": X["customer_id"].astype(int), "propensity_investment": p_all})
    scores_path = Path("reports/tables/propensity_scores.csv")
    scores.to_csv(scores_path, index=False)

    # Save model (optional)
    if joblib is not None:
        joblib.dump(clf, Path("models/propensity_model.pkl"))

    print(f"Saved: {metrics_path} | auc={auc:.3f} ks={ks:.3f}")
    print(f"Saved: {scores_path} | rows={len(scores):,}")
    if joblib is not None:
        print("Saved: models/propensity_model.pkl")


if __name__ == "__main__":
    main()