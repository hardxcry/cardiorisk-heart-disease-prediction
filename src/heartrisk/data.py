"""Download and clean the UCI Heart Disease dataset (all four collection sites).

Source: Janosi, Steinbrunn, Pfisterer, Detrano (1989), UCI Machine Learning Repository,
https://doi.org/10.24432/C52P4X  (CC BY 4.0)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
SITES = {
    "cleveland": "processed.cleveland.data",
    "hungary": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va_long_beach": "processed.va.data",
}
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "disease"

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (1 = male)",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy",
    "thal": "Thallium stress test result",
}


def download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fname in SITES.values():
        out = raw_dir / fname
        if out.exists():
            continue
        r = requests.get(BASE + fname, timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)


def load(raw_dir: Path) -> pd.DataFrame:
    """Return one cleaned table with a `site` column and a binary `disease` target."""
    download(raw_dir)
    frames = []
    for site, fname in SITES.items():
        df = pd.read_csv(raw_dir / fname, header=None, names=COLUMNS, na_values="?")
        df["site"] = site
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Physiologically impossible zeros are missing-value codes in the source files.
    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    df.loc[df["oldpeak"] < 0, "oldpeak"] = np.nan

    # Original label is 0-4; the standard task is presence of any disease.
    df[TARGET] = (df["num"] > 0).astype(int)
    return df.drop(columns="num")


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    return (df[FEATURES].isna().groupby(df["site"]).mean() * 100).round(1).T
