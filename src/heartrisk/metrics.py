"""Clinical evaluation metrics: discrimination, calibration, clinical utility."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score, roc_curve)


def threshold_for_sensitivity(y, p, target: float = 0.90) -> float:
    """Highest threshold whose sensitivity is at least `target` (rule-out use)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return float(thr[ok[0]])


def confusion_stats(y, p, threshold: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else float("nan")
    return {
        "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Logistic recalibration: ideal slope 1, intercept 0."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def summary(y, p, threshold: float) -> dict:
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "n": int(len(y)), "prevalence": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_slope": slope, "calibration_intercept": intercept,
        "threshold": threshold, **confusion_stats(y, p, threshold),
    }


def bootstrap_ci(y, p, threshold: float, n_boot: int = 2000, seed: int = 42) -> dict:
    """Percentile 95% confidence intervals from patient-level resampling."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    keys = ["auroc", "auprc", "brier", "sensitivity", "specificity", "ppv", "npv"]
    rows = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        yb, pb = y[idx], p[idx]
        rows["auroc"].append(roc_auc_score(yb, pb))
        rows["auprc"].append(average_precision_score(yb, pb))
        rows["brier"].append(brier_score_loss(yb, pb))
        cs = confusion_stats(yb, pb, threshold)
        for k in keys[3:]:
            rows[k].append(cs[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in rows.items()}


def net_benefit(y, p, thresholds) -> dict:
    """Decision-curve analysis (Vickers & Elkin 2006)."""
    y, p = np.asarray(y), np.asarray(p)
    n, prev = len(y), np.mean(y)
    model, treat_all = [], []
    for t in thresholds:
        pred = p >= t
        tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
        w = t / (1 - t)
        model.append(float(tp / n - fp / n * w))
        treat_all.append(float(prev - (1 - prev) * w))
    return {"thresholds": [float(t) for t in thresholds], "model": model, "treat_all": treat_all}
