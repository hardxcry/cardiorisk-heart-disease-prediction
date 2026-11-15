"""End-to-end experiment: internal CV, external (leave-one-site-out) validation,
calibration, decision curves, subgroup audit, explainability, final model export.

Run:  python -m heartrisk.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from . import metrics as M
from .data import FEATURE_LABELS, FEATURES, SITES, TARGET, load, missingness
from .models import SEED, candidates

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports" / "figures"
plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})


def oof_predictions(model, X, y, n_splits=5, n_repeats=5) -> np.ndarray:
    """Out-of-fold probabilities averaged over repeated stratified CV."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    total = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        total[te] += m.predict_proba(X.iloc[te])[:, 1]
    return total / n_repeats


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    df = load(ROOT / "data" / "raw")
    X, y = df[FEATURES], df[TARGET]
    out: dict = {
        "dataset": {
            "n": int(len(df)), "prevalence": float(y.mean()),
            "per_site": {s: {"n": int((df.site == s).sum()),
                             "prevalence": float(y[df.site == s].mean())} for s in SITES},
            "missing_pct": missingness(df).to_dict(),
        },
        "internal_cv": {}, "external_loso": {},
    }

    # 1) Internal validation: 5x5 repeated stratified CV on pooled data ----------
    oof = {}
    for name, model in candidates().items():
        p = oof_predictions(model, X, y)
        thr = M.threshold_for_sensitivity(y, p, 0.90)
        oof[name] = p
        out["internal_cv"][name] = {**M.summary(y, p, thr), "ci95": M.bootstrap_ci(y, p, thr)}
        print(f"[internal] {name}: AUROC {out['internal_cv'][name]['auroc']:.3f}")
    best = max(out["internal_cv"], key=lambda k: out["internal_cv"][k]["auroc"])
    out["selected_model"] = best
    p_best, thr_best = oof[best], out["internal_cv"][best]["threshold"]

    # 2) External validation: train on three hospitals, test on the fourth -------
    for name, model in candidates().items():
        out["external_loso"][name] = {}
        for site in SITES:
            tr, te = df.site != site, df.site == site
            m = clone(model).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            out["external_loso"][name][site] = M.summary(y[te], p, thr_best)
        aucs = [v["auroc"] for v in out["external_loso"][name].values()]
        print(f"[external] {name}: mean AUROC {np.mean(aucs):.3f}")

    # 3) Subgroup audit on the selected model ------------------------------------
    out["subgroups"] = {}
    groups = {"female": df.sex == 0, "male": df.sex == 1,
              "age_under_55": df.age < 55, "age_55_plus": df.age >= 55}
    for g, mask in groups.items():
        out["subgroups"][g] = M.summary(y[mask], p_best[mask], thr_best)

    # 4) Figures -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4.6))
    for name, p in oof.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUROC {out['internal_cv'][name]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="1 - specificity", ylabel="Sensitivity", title="ROC, repeated 5-fold CV")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "roc.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.6))
    for name, p in oof.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", lw=1.5, ms=4, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="Predicted risk", ylabel="Observed frequency", title="Calibration")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "calibration.png"); plt.close(fig)

    nb = M.net_benefit(y, p_best, np.linspace(0.05, 0.80, 76))
    out["decision_curve"] = nb
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot(nb["thresholds"], nb["model"], lw=2, label=best)
    ax.plot(nb["thresholds"], nb["treat_all"], lw=1.5, label="Refer everyone")
    ax.axhline(0, color="k", lw=1, label="Refer no one")
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", title="Decision curve",
           ylim=(-0.05, max(nb["model"]) + 0.05))
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "decision_curve.png"); plt.close(fig)

    # 5) Final model on all data + global importance ------------------------------
    final = clone(candidates()[best]).fit(X, y)
    imp = permutation_importance(final, X, y, scoring="roc_auc", n_repeats=30, random_state=SEED)
    order = np.argsort(imp.importances_mean)
    out["permutation_importance"] = {FEATURES[i]: float(imp.importances_mean[i]) for i in order[::-1]}
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh([FEATURE_LABELS[FEATURES[i]] for i in order], imp.importances_mean[order],
            xerr=imp.importances_std[order], color="#c0392b")
    ax.set(xlabel="Drop in AUROC when feature is shuffled", title="Global feature importance")
    fig.tight_layout(); fig.savefig(FIG / "importance.png"); plt.close(fig)

    joblib.dump({"model": final, "threshold": thr_best, "features": FEATURES, "name": best,
                 "background": X.sample(100, random_state=SEED).reset_index(drop=True),
                 "medians": X.median().to_dict()}, ROOT / "models" / "model.joblib")
    df.to_csv(ROOT / "data" / "heart_clean.csv", index=False)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(out, indent=2))
    print("selected:", best, "| threshold:", round(thr_best, 3))


if __name__ == "__main__":
    main()
