"""
LA Anesthetic Success Predictor — Streamlit Web App
Prof. Vivek Aggarwal, JMI New Delhi
Department of Conservative Dentistry & Endodontics

This app trains a Logistic Regression model on your dataset at startup,
then allows per-patient prediction of anesthetic success probability.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LA Success Predictor",
    page_icon="🦷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a3c5e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 1rem;
        font-weight: 600;
    }
    .high-success    { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .moderate        { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .low-prob        { background: #ffe5b4; color: #7d4e00; border: 2px solid #fd7e14; }
    .very-high-risk  { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    .metric-row {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1a3c5e;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🦷 LA Anesthetic Success Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Endodontic Local Anesthesia — AI-Based Clinical Decision Support<br>Prof. Vivek Aggarwal · JMI New Delhi</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR — DATASET UPLOAD
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    st.markdown("**Upload your dataset to train the model:**")
    uploaded_file = st.file_uploader(
        "LA_Endodontic_Dataset_4390.xlsx",
        type=["xlsx"],
        help="Must contain columns: Age, HP_VAS_Score, Gender, Alcohol_Use, Tooth_Type, Preop_Medication, Pain_Intensity, Age_Group, Anesthesia_Success"
    )

    st.markdown("---")
    st.markdown("**Required columns:**")
    st.code("""Age
HP_VAS_Score
Gender
Alcohol_Use
Tooth_Type
Preop_Medication
Pain_Intensity
Age_Group
Anesthesia_Success""", language=None)

    st.markdown("---")
    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Split:** 70/15/15 stratified")
    st.markdown("**Developer:** Prof. Vivek Aggarwal")

# ─────────────────────────────────────────────
# SYNTHETIC DATA FALLBACK (mirrors your dataset statistics)
# ─────────────────────────────────────────────
@st.cache_data
def generate_synthetic_data():
    """Generate synthetic dataset matching published statistics for demo."""
    np.random.seed(42)
    n = 4390

    tooth_types = [
        "Maxillary Incisors/Canine",
        "Mandibular Premolars",
        "Maxillary Premolars",
        "Maxillary Molars",
        "Mandibular Anteriors",
        "Mandibular Molars"
    ]
    tooth_n      = [287, 887, 798, 1017, 223, 1178]
    tooth_success = [0.882, 0.861, 0.846, 0.806, 0.605, 0.392]

    rows = []
    for tt, nt, sr in zip(tooth_types, tooth_n, tooth_success):
        for _ in range(nt):
            success = int(np.random.rand() < sr)
            age = int(np.random.normal(38, 12))
            age = max(18, min(70, age))
            if age <= 35:   ag = "18-35"
            elif age <= 55: ag = "36-55"
            else:           ag = "56-70"

            vas = int(np.random.normal(90 if not success else 70, 25))
            vas = max(0, min(170, vas))

            pi = "Severe" if vas > 114 else ("Moderate" if vas > 54 else "Mild")
            alcohol = "Yes" if (np.random.rand() < (0.35 if not success else 0.20)) else "No"
            preop   = "Yes" if np.random.rand() < 0.40 else "No"
            gender  = "Female" if np.random.rand() < 0.48 else "Male"

            rows.append({
                "Age": age, "HP_VAS_Score": vas, "Gender": gender,
                "Alcohol_Use": alcohol, "Tooth_Type": tt,
                "Preop_Medication": preop, "Pain_Intensity": pi,
                "Age_Group": ag, "Anesthesia_Success": success
            })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_model(data_key):
    """Train logistic regression. data_key changes when new file uploaded."""
    return _train(data_key)

def _train(df):
    feature_cols = ["Age", "HP_VAS_Score", "Gender", "Alcohol_Use",
                    "Tooth_Type", "Preop_Medication", "Pain_Intensity", "Age_Group"]

    df_enc = pd.get_dummies(df[feature_cols + ["Anesthesia_Success"]],
                            columns=["Gender", "Alcohol_Use", "Tooth_Type",
                                     "Preop_Medication", "Pain_Intensity", "Age_Group"])

    X = df_enc.drop("Anesthesia_Success", axis=1)
    y = df_enc["Anesthesia_Success"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30,
                                                         stratify=y, random_state=42)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50,
                                                         stratify=y_temp, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train_s, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])

    return model, scaler, list(X.columns), round(auc, 3)

# ─────────────────────────────────────────────
# LOAD DATA & TRAIN
# ─────────────────────────────────────────────
if uploaded_file:
    df_raw = pd.read_excel(uploaded_file)
    data_source = "uploaded"
    st.sidebar.success(f"✅ Loaded {len(df_raw):,} rows from your file")
else:
    df_raw = generate_synthetic_data()
    data_source = "synthetic"
    st.info("ℹ️ No dataset uploaded — using **synthetic demo data** that mirrors your published statistics. Upload `LA_Endodontic_Dataset_4390.xlsx` in the sidebar to use your real model.", icon="📂")

with st.spinner("Training model…"):
    model, scaler, feature_cols, auc = _train(df_raw)

st.success(f"✅ Model ready — AUC-ROC: **{auc}** ({'your real data' if data_source == 'uploaded' else 'synthetic demo'})")

# ─────────────────────────────────────────────
# PATIENT INPUT FORM
# ─────────────────────────────────────────────
st.markdown("## 📋 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=70, value=35, step=1)
    hp_vas = st.slider("HP-VAS Score (mm)", min_value=0, max_value=170, value=80,
                       help="Heft-Parker Visual Analogue Scale: 0=no pain, 170=maximum pain")
    gender = st.selectbox("Gender", ["Male", "Female"])
    alcohol = st.selectbox("Alcohol Use", ["No", "Yes"])

with col2:
    tooth_type = st.selectbox("Tooth Type", [
        "Maxillary Incisors/Canine",
        "Mandibular Premolars",
        "Maxillary Premolars",
        "Maxillary Molars",
        "Mandibular Anteriors",
        "Mandibular Molars"
    ])
    preop_med = st.selectbox("Pre-operative Medication", ["No", "Yes"])
    pain_intensity = st.selectbox("Pain Intensity", ["Mild", "Moderate", "Severe"],
                                  index=1)
    if age <= 35:   age_group = "18-35"
    elif age <= 55: age_group = "36-55"
    else:           age_group = "56-70"
    st.info(f"Age Group (auto): **{age_group}**")

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict(age, hp_vas, gender, alcohol, tooth_type, preop_med, pain_intensity, age_group):
    patient = {
        "Age": age, "HP_VAS_Score": hp_vas,
        f"Gender_{gender}": 1,
        f"Alcohol_Use_{alcohol}": 1,
        f"Tooth_Type_{tooth_type}": 1,
        f"Preop_Medication_{preop_med}": 1,
        f"Pain_Intensity_{pain_intensity}": 1,
        f"Age_Group_{age_group}": 1
    }
    row = pd.DataFrame([{col: patient.get(col, 0) for col in feature_cols}])
    row_s = scaler.transform(row)
    prob = model.predict_proba(row_s)[0][1]
    return prob

st.markdown("---")

if st.button("🔍 Predict Anesthetic Success", use_container_width=True, type="primary"):
    prob = predict(age, hp_vas, gender, alcohol, tooth_type, preop_med, pain_intensity, age_group)
    pct  = round(prob * 100, 1)

    st.markdown("## 🎯 Prediction Result")

    # Progress bar
    st.progress(prob)
    st.markdown(f"### Predicted Success Probability: **{pct}%**")

    # Risk band
    if prob >= 0.80:
        band = "High Success"
        css  = "high-success"
        rec  = "✅ Proceed with standard IANB protocol"
        icon = "🟢"
    elif prob >= 0.60:
        band = "Moderate Probability"
        css  = "moderate"
        rec  = "⚠️ Consider supplemental anaesthesia techniques (intraligamentary / intrapulpal)"
        icon = "🟡"
    elif prob >= 0.40:
        band = "Low Probability"
        css  = "low-prob"
        rec  = "🔶 Plan supplemental anaesthesia before starting treatment"
        icon = "🟠"
    else:
        band = "Very High Risk of Failure"
        css  = "very-high-risk"
        rec  = "🔴 Strongly consider alternative approach — Gow-Gates / Vazirani-Akinosi / intraosseous"
        icon = "🔴"

    st.markdown(f'<div class="result-box {css}">{icon} {band}<br><small>{rec}</small></div>',
                unsafe_allow_html=True)

    # Summary table
    st.markdown("### 📊 Patient Summary")
    summary = pd.DataFrame({
        "Parameter": ["Tooth Type", "HP-VAS Score", "Pain Intensity", "Alcohol Use",
                      "Pre-op Medication", "Age", "Gender", "Success Probability", "Risk Band"],
        "Value": [tooth_type, f"{hp_vas} mm", pain_intensity, alcohol,
                  preop_med, f"{age} yrs ({age_group})", gender, f"{pct}%", band]
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

    # Contextual note
    st.markdown("---")
    st.caption(
        "⚕️ This tool provides decision support only and does not replace clinical judgement. "
        "Model trained on data from the Department of Conservative Dentistry & Endodontics, JMI New Delhi. "
        "AUC-ROC: " + str(auc) + ". For research and educational use."
    )

# ─────────────────────────────────────────────
# DATASET STATS (expandable)
# ─────────────────────────────────────────────
with st.expander("📈 Dataset & Model Statistics"):
    st.markdown(f"**Rows:** {len(df_raw):,} &nbsp;|&nbsp; **Model AUC-ROC:** {auc} &nbsp;|&nbsp; **Data source:** {'Uploaded file' if data_source == 'uploaded' else 'Synthetic demo'}")

    tooth_stats = df_raw.groupby("Tooth_Type")["Anesthesia_Success"].agg(
        N="count", Success_Rate="mean"
    ).reset_index()
    tooth_stats["Success_Rate"] = (tooth_stats["Success_Rate"] * 100).round(1).astype(str) + "%"
    tooth_stats = tooth_stats.sort_values("N", ascending=False)
    st.dataframe(tooth_stats, hide_index=True, use_container_width=True)
