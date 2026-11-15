"""CardioRisk: interactive demo of the coronary artery disease risk model.

Run:  streamlit run app/streamlit_app.py
Research prototype. Not a medical device and not for clinical use.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="CardioRisk", page_icon="🫀", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg,#7f1d1d 0%,#b91c1c 55%,#ef4444 100%);
       padding: 1.6rem 2rem; border-radius: 18px; color: #fff; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.hero p {margin: .3rem 0 0; opacity: .9;}
.card {background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.2);
       border-radius: 14px; padding: 1rem 1.2rem;}
.pill {display:inline-block; padding:.25rem .8rem; border-radius:999px; font-weight:600;}
.hi {background:#fee2e2; color:#991b1b;} .lo {background:#dcfce7; color:#166534;}
.small {font-size:.85rem; opacity:.75;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "model.joblib")


@st.cache_data
def load_metrics():
    return json.loads((ROOT / "reports" / "metrics.json").read_text())


bundle, metrics = load_bundle(), load_metrics()
model, threshold, features = bundle["model"], bundle["threshold"], bundle["features"]

LABELS = {"age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Resting BP",
          "chol": "Cholesterol", "fbs": "Fasting blood sugar", "restecg": "Resting ECG",
          "thalach": "Max heart rate", "exang": "Exercise angina", "oldpeak": "ST depression",
          "slope": "ST slope", "ca": "Vessels (fluoroscopy)", "thal": "Thallium test"}
NA = "Not measured"


def pick(label, options: dict, default=0, key=None):
    choice = st.selectbox(label, list(options), index=default, key=key)
    return options[choice]


st.markdown('<div class="hero"><h1>🫀 CardioRisk</h1><p>Explainable coronary artery disease '
            'risk estimation, validated across four hospitals.</p></div>', unsafe_allow_html=True)
tab_patient, tab_perf, tab_about = st.tabs(["Patient assessment", "Model performance", "About"])

with tab_patient:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("Clinical inputs")
        a, b = st.columns(2)
        with a:
            age = st.slider("Age (years)", 25, 85, 55)
            sex = pick("Sex", {"Male": 1, "Female": 0})
            trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 200, 130)
            chol_known = st.toggle("Cholesterol measured", True)
            chol = st.slider("Serum cholesterol (mg/dl)", 100, 600, 240) if chol_known else np.nan
            fbs = pick("Fasting blood sugar > 120 mg/dl", {"No": 0, "Yes": 1})
            thalach = st.slider("Max heart rate achieved", 60, 210, 150)
        with b:
            cp = pick("Chest pain type", {"Typical angina": 1, "Atypical angina": 2,
                                          "Non-anginal pain": 3, "Asymptomatic": 4}, 3)
            restecg = pick("Resting ECG", {"Normal": 0, "ST-T abnormality": 1,
                                           "LV hypertrophy": 2})
            exang = pick("Exercise-induced angina", {"No": 0, "Yes": 1})
            oldpeak = st.slider("ST depression (mm)", 0.0, 6.0, 1.0, 0.1)
            slope = pick("ST slope", {NA: np.nan, "Upsloping": 1, "Flat": 2, "Downsloping": 3}, 2)
            ca = pick("Vessels on fluoroscopy", {NA: np.nan, "0": 0, "1": 1, "2": 2, "3": 3})
            thal = pick("Thallium stress test", {NA: np.nan, "Normal": 3, "Fixed defect": 6,
                                                 "Reversible defect": 7})
    row = pd.DataFrame([{"age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
                         "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
                         "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal}])[features]
    row = row.astype(float)
    risk = float(model.predict_proba(row)[0, 1])
    high = risk >= threshold

    with right:
        st.subheader("Estimated risk")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=risk * 100, number={"suffix": "%", "font": {"size": 54}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#b91c1c" if high else "#15803d"},
                   "steps": [{"range": [0, threshold * 100], "color": "rgba(34,197,94,.18)"},
                             {"range": [threshold * 100, 100], "color": "rgba(239,68,68,.18)"}],
                   "threshold": {"line": {"color": "#111", "width": 3}, "value": threshold * 100}}))
        gauge.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=0),
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True)
        tag = ('<span class="pill hi">Above referral threshold</span>' if high
               else '<span class="pill lo">Below referral threshold</span>')
        st.markdown(f'{tag} <span class="small">&nbsp;Threshold {threshold:.0%}, set for 90% '
                    f'sensitivity in validation.</span>', unsafe_allow_html=True)

        st.subheader("Why this estimate")
        try:
            import shap
            explainer = shap.Explainer(lambda d: model.predict_proba(
                pd.DataFrame(d, columns=features))[:, 1], bundle["background"].astype(float))
            sv = explainer(row, max_evals=2 * len(features) + 1, silent=True)
            contrib = pd.Series(sv.values[0], index=[LABELS[f] for f in features]).sort_values()
            bar = go.Figure(go.Bar(x=contrib.values * 100, y=contrib.index, orientation="h",
                                   marker_color=["#b91c1c" if v > 0 else "#15803d"
                                                 for v in contrib.values]))
            bar.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=30),
                              xaxis_title="Change in risk (percentage points) vs average patient",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(bar, use_container_width=True)
            st.caption("SHAP values: red raises this patient's risk, green lowers it.")
        except Exception as exc:  # explanation is optional; the estimate still stands
            st.info(f"Explanation unavailable on this machine ({type(exc).__name__}).")

with tab_perf:
    name = metrics["selected_model"]
    m = metrics["internal_cv"][name]
    ext = metrics["external_loso"][name]
    ext_auc = float(np.mean([v["auroc"] for v in ext.values()]))
    c = st.columns(4)
    c[0].metric("AUROC (internal CV)", f"{m['auroc']:.3f}",
                help=f"95% CI {m['ci95']['auroc'][0]:.3f} to {m['ci95']['auroc'][1]:.3f}")
    c[1].metric("AUROC (unseen hospital)", f"{ext_auc:.3f}", f"{ext_auc - m['auroc']:+.3f}")
    c[2].metric("Sensitivity", f"{m['sensitivity']:.1%}")
    c[3].metric("Specificity", f"{m['specificity']:.1%}")
    st.caption(f"Selected model: {name.replace('_', ' ')}. {metrics['dataset']['n']} patients, "
               f"four hospitals. The drop on unseen hospitals is the honest number.")
    f = st.columns(2)
    f[0].image(str(ROOT / "reports/figures/roc.png"))
    f[1].image(str(ROOT / "reports/figures/calibration.png"))
    f = st.columns(2)
    f[0].image(str(ROOT / "reports/figures/decision_curve.png"))
    f[1].image(str(ROOT / "reports/figures/importance.png"))
    st.subheader("Leave-one-hospital-out validation")
    st.dataframe(pd.DataFrame(ext).T[["n", "prevalence", "auroc", "sensitivity", "specificity",
                                      "brier"]].round(3), use_container_width=True)
    st.subheader("Subgroup audit")
    st.dataframe(pd.DataFrame(metrics["subgroups"]).T[["n", "prevalence", "auroc", "sensitivity",
                                                       "specificity"]].round(3),
                 use_container_width=True)

with tab_about:
    st.markdown("""
**What this is.** A research prototype that estimates the probability of angiographic coronary
artery disease from 13 routine clinical variables, trained on the public UCI Heart Disease
dataset (Cleveland, Budapest, Zurich/Basel, Long Beach VA).

**What it is not.** It is not a medical device, has not been prospectively validated, and must
not be used to make decisions about real patients.

**Known limitations.** Data were collected in the 1980s from patients already referred for
angiography, so disease prevalence is far higher than in primary care. Performance falls on
hospitals the model has not seen. Women are under-represented.

See `reports/REPORT.md` and `MODEL_CARD.md` in the repository for the full methodology.
""")
