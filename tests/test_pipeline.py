import numpy as np
import pandas as pd
import pytest

from heartrisk import metrics as M
from heartrisk.data import CATEGORICAL, FEATURES, NUMERIC
from heartrisk.models import candidates


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({f: rng.normal(size=n) for f in NUMERIC} |
                     {f: rng.integers(0, 3, n).astype(float) for f in CATEGORICAL})[FEATURES]
    y = pd.Series((X["age"] + X["oldpeak"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    X.loc[rng.random(n) < 0.2, "chol"] = np.nan  # models must tolerate missing values
    X.loc[rng.random(n) < 0.2, "thal"] = np.nan
    return X, y


@pytest.mark.parametrize("name", list(candidates()))
def test_models_fit_with_missing_values(toy, name):
    X, y = toy
    p = candidates()[name].fit(X, y).predict_proba(X)[:, 1]
    assert p.shape == (len(y),) and np.all((p >= 0) & (p <= 1))
    assert M.summary(y, p, 0.5)["auroc"] > 0.8


def test_threshold_reaches_target_sensitivity():
    y = np.array([0] * 50 + [1] * 50)
    p = np.linspace(0, 1, 100)
    t = M.threshold_for_sensitivity(y, p, 0.90)
    assert M.confusion_stats(y, p, t)["sensitivity"] >= 0.90


def test_confusion_stats_known_case():
    s = M.confusion_stats(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
    assert (s["tp"], s["fn"], s["fp"], s["tn"]) == (1, 1, 1, 1)
    assert s["sensitivity"] == 0.5 and s["specificity"] == 0.5


def test_net_benefit_of_perfect_model_equals_prevalence():
    y = np.array([0, 0, 1, 1])
    nb = M.net_benefit(y, y.astype(float), [0.5])
    assert nb["model"][0] == pytest.approx(0.5)


def test_perfect_calibration_slope_near_one():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    slope, intercept = M.calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.08) and intercept == pytest.approx(0.0, abs=0.08)
