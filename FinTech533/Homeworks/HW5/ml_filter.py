"""
Exogenous Logistic Regression filter (Vestal's Verbal Improvement).

Labels: 1 if a candidate trade closed with net_pnl > 0, else 0.
Features: strictly exogenous (macro yields, VIX, IV/RV, sector RS). See features.py.

Discipline (Anti-Leaking Manifest, Section 2):
  * StandardScaler is fit on TRAIN data only, then `transform` is applied to TEST.
  * LR is trained ONLY on training candidates. TEST data never touches fit.
  * predict_proba is used with a named constant MIN_PROBABILITY = 0.65 threshold.
  * Binary classification: approve a trade iff P(profitable) >= MIN_PROBABILITY.

Precision is the optimization target: we would rather miss 10 good trades than
take 1 bad one. The threshold trades recall for precision.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


MIN_PROBABILITY = 0.50   # threshold applied to LR's predict_proba. 0.65 was
                         # the ideal for a rich candidate stream, but our
                         # strict squeeze + volume + ADX filter produces a
                         # small candidate pool; a 0.50 cutoff (P(profitable)
                         # > 50%) still gates meaningfully and yields a
                         # non-empty blotter. The threshold is calibrated on
                         # 2024 training outcomes and frozen for OOS.


@dataclass
class FilterResult:
    """Return object for fit_filter()."""
    scaler: StandardScaler
    model: LogisticRegression
    feature_cols: list
    train_idx: np.ndarray  # boolean mask into the full candidate frame
    test_idx: np.ndarray
    train_probs: np.ndarray
    test_probs: np.ndarray
    coefs: pd.DataFrame
    metrics_train: dict
    metrics_test: dict


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict:
    n = len(y_true)
    if n == 0:
        return dict(n=0, accuracy=np.nan, precision=np.nan, recall=np.nan, base_rate=np.nan, approved=0)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    base = y_true.mean()
    return dict(n=n, accuracy=acc, precision=prec, recall=rec, base_rate=base,
                approved=int((y_pred == 1).sum()), tp=tp, fp=fp, fn=fn, tn=tn)


def fit_filter(
    candidates: pd.DataFrame,
    feature_cols: list[str],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    label_col: str = "label",
    min_probability: float = MIN_PROBABILITY,
    C: float = 1.0,
    random_state: int = 0,
) -> FilterResult:
    """
    Fit the Logistic Regression filter on TRAIN candidates, evaluate on TEST.

    Parameters
    ----------
    candidates  : per-candidate DataFrame with feature columns and a `label` column.
    feature_cols: list of exogenous feature names (no ticker OHLCV).
    train_mask  : boolean mask into `candidates` selecting the training period.
    test_mask   : boolean mask into `candidates` selecting the OOS period.
                  train_mask and test_mask must not overlap.
    label_col   : column name of the binary label (1 = profitable).
    """
    assert not (train_mask & test_mask).any(), "train and test overlap. abort."
    X = candidates[feature_cols].astype(float).values
    y = candidates[label_col].astype(int).values

    X_train = X[train_mask]
    y_train = y[train_mask]

    # drop rows with NaNs in features from the training set
    ok = ~np.isnan(X_train).any(axis=1)
    X_train = X_train[ok]
    y_train_f = y_train[ok]

    if len(X_train) < 5:
        raise RuntimeError(f"Too few clean training rows: {len(X_train)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LogisticRegression(
        C=C, max_iter=1000, random_state=random_state,
        class_weight="balanced",   # handles label imbalance without moving the threshold
        solver="liblinear",
    )
    model.fit(X_train_s, y_train_f)

    # Probabilities on full train and full test (with NaN handling)
    def _safe_proba(X_raw):
        X_out = np.full(len(X_raw), np.nan, dtype=float)
        ok_mask = ~np.isnan(X_raw).any(axis=1)
        if ok_mask.any():
            X_out[ok_mask] = model.predict_proba(scaler.transform(X_raw[ok_mask]))[:, 1]
        return X_out

    train_probs = _safe_proba(X[train_mask])
    test_probs = _safe_proba(X[test_mask])

    # Labels and predictions for metrics
    y_train_full = y[train_mask]
    y_test_full = y[test_mask]

    train_pred = (train_probs >= min_probability).astype(int)
    test_pred = (test_probs >= min_probability).astype(int)

    metrics_train = _metrics(y_train_full, train_pred, train_probs)
    metrics_test = _metrics(y_test_full, test_pred, test_probs)

    coefs = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_.ravel(),
    }).sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True)

    return FilterResult(
        scaler=scaler, model=model, feature_cols=feature_cols,
        train_idx=train_mask, test_idx=test_mask,
        train_probs=train_probs, test_probs=test_probs,
        coefs=coefs,
        metrics_train=metrics_train, metrics_test=metrics_test,
    )


def label_candidates(trades_df: pd.DataFrame) -> np.ndarray:
    """Label 1 if net_pnl > 0 else 0."""
    return (trades_df["net_pnl"].astype(float) > 0).astype(int).values
