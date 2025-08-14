# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="Drill String Failure Prediction", layout="wide")

# --------------------
# Load model & artifacts
# --------------------
ARTIFACT_PKL = "failure_prediction_ann.pkl"   # produced by your training script
DATASET_PATH = "physical_drill_string_dataset_with_risk.csv"  # optional - used to compute defaults

@st.cache_resource
def load_artifacts():
    with open(ARTIFACT_PKL, "rb") as f:
        artifacts = pickle.load(f)
    model = tf.keras.models.load_model(artifacts["model_path"])
    preprocessor = artifacts["preprocessor"]
    feature_names = artifacts["features"]
    return model, preprocessor, feature_names

try:
    model, preprocessor, feature_names = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts ({ARTIFACT_PKL} or model path). Make sure the files exist. \n{e}")
    st.stop()

# --------------------
# Attempt to compute sensible defaults (medians) from dataset if available
# --------------------
def compute_defaults(path, feature_names):
    try:
        df = pd.read_csv(path)
        defaults = {}
        for c in feature_names:
            if c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    defaults[c] = float(df[c].median(skipna=True))
                else:
                    # choose most common string or first unique (fallback)
                    vals = df[c].dropna().unique()
                    defaults[c] = str(vals[0]) if len(vals)>0 else ""
            else:
                defaults[c] = 0.0 if isinstance(0.0, (int,float)) else ""
        return defaults
    except Exception:
        # fallback: zeros and some common string labels
        fallback = {c: 0.0 for c in feature_names}
        return fallback

defaults = compute_defaults(DATASET_PATH, feature_names)

# --------------------
# Helpful mappings (human-facing)
# --------------------
# These should match the string categories present in your original dataset for 'Bit Type' and 'Formation Type'.
# If different in your data, adjust the lists below to match exact dataset strings.
BIT_TYPE_OPTIONS = ["PDC", "Roller Cone", "Diamond", "Other"]
FORMATION_TYPE_OPTIONS = ["Sandstone", "Shale", "Limestone", "Salt", "Other"]

# --------------------
# Thresholds & recommendation templates
# --------------------
THRESHOLDS = {
    'Rotary Speed (RPM)': {'warning': 150, 'critical': 180, 'impact': "Excessive RPM causes fatigue and vibration damage"},
    'Weight on Bit (WOB) (lbs)': {'warning': 35000, 'critical': 40000, 'impact': "High WOB leads to buckling and twist-off risk"},
    'Torque (ft-lbs)': {'warning': 18000, 'critical': 20000, 'impact': "Over-torque can cause tool joint failure"},
    'Vibration_Level': {'warning': 3, 'critical': 5, 'impact': "Vibration accelerates fatigue and BHA damage"},
    'Von Mises stress (psi)': {'warning': 70000, 'critical': 80000, 'impact': "High stress causes plastic deformation"},
    'Bottom hole temperature (°F)': {'warning': 250, 'critical': 300, 'impact': "Heat reduces material strength"},
    'Dogleg severity (deg/100ft)': {'warning': 5, 'critical': 8, 'impact': "Sharp bends cause cyclic stress"},
    'Flow Rate (gpm)': {'warning': 600, 'critical': 800, 'impact': "High flow erodes tools and causes vibration"},
    'Mud weight (ppg)': {'warning': 14, 'critical': 16, 'impact': "Dense mud increases pressure differential"},
    'Yield strength (psi)': {'warning': 250000, 'critical': 500000, 'impact': "Low yield strength risks deformation"},
    'Toughness (ft-lbs)': {'warning': 40, 'critical': 20, 'impact': "Low toughness increases crack propagation risk"},
    'Hardness (HRC)': {'warning': 35, 'critical': 45, 'impact': "High hardness reduces toughness"},
    # add others if you have them
}

def analyze_parameters(input_dict):
    exceeded = []
    for param, levels in THRESHOLDS.items():
        if param in input_dict:
            try:
                value = float(input_dict[param])
            except Exception:
                continue
            if value >= levels['critical']:
                exceeded.append((param, value, levels['critical'], 'critical', levels['impact']))
            elif value >= levels['warning']:
                exceeded.append((param, value, levels['warning'], 'warning', levels['impact']))
    return exceeded

def generate_recommendations(exceeded_params, proba, input_dict):
    recommendations = {
        'Immediate Actions': [],
        'Maintenance Recommendations': [],
        'Design Improvements': [],
        'Monitoring Suggestions': [],
        'String Configuration Adjustments': []
    }

    if proba >= 0.7:
        recommendations['Immediate Actions'].append("Reduce drilling parameters immediately and perform inspection")
        recommendations['Monitoring Suggestions'].append("Enable continuous real-time monitoring")
    elif proba >= 0.5:
        recommendations['Maintenance Recommendations'].append("Schedule inspection at next available connection")
        recommendations['Monitoring Suggestions'].append("Increase monitoring frequency")

    for param, value, threshold, severity, impact in exceeded_params:
        if param == 'Rotary Speed (RPM)':
            recommendations['Immediate Actions'].append(f"Reduce RPM below {threshold} (current {value})")
            recommendations['Design Improvements'].append("Consider vibration dampeners in BHA")
        elif param in ['Weight on Bit (WOB) (lbs)']:
            recommendations['Immediate Actions'].append(f"Reduce WOB to below {threshold} lbs (current {value})")
            recommendations['Design Improvements'].append("Optimize bit selection for formation")
        elif param == 'Vibration_Level':
            recommendations['Immediate Actions'].append(f"Address vibration (Level {value}) - reduce RPM / check BHA")
            recommendations['Design Improvements'].append("Install shock subs or vibration dampeners")
            recommendations['Maintenance Recommendations'].append("Inspect tool joints for fatigue")
        elif param == 'Von Mises stress (psi)':
            recommendations['Immediate Actions'].append("Reduce torsional/bending loads")
            recommendations['Maintenance Recommendations'].append("Perform non-destructive testing for cracks")
            recommendations['Design Improvements'].append("Consider higher grade materials")
        elif param == 'Bottom hole temperature (°F)':
            recommendations['Immediate Actions'].append("Increase circulation/monitor mud cooling")
            recommendations['Design Improvements'].append("Consider HT material upgrade")
        # generic fallback
        else:
            recommendations['Monitoring Suggestions'].append(f"Monitor {param} closely (current {value})")

    return recommendations

# --------------------
# UI layout - input sections
# --------------------
st.title("🚨 Drill String Failure Prediction")
st.write("Fill the sections below. If you don't know a value, leave the default (dataset median when available).")

# Use two-column layout for inputs
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Material Parameters")
    bit_type = st.selectbox("Bit Type", BIT_TYPE_OPTIONS)
    formation_type = st.selectbox("Formation Type", FORMATION_TYPE_OPTIONS)
    mud_weight_ppg = st.number_input("Mud weight (ppg)", value=float(defaults.get("Mud weight (ppg)", 10.0)))
    drill_pipe_weight = st.number_input("Drill pipe weight (lbs/ft)", value=float(defaults.get("Drill pipe weight (lbs/ft)", 100.0)))
    yield_strength = st.number_input("Yield strength (psi)", value=float(defaults.get("Yield strength (psi)", 135000.0)))
    toughness = st.number_input("Toughness (ft-lbs)", value=float(defaults.get("Toughness (ft-lbs)", 80.0)))
    hardness = st.number_input("Hardness (HRC)", value=float(defaults.get("Hardness (HRC)", 22.0)))

with col_right:
    st.subheader("Dimensional Parameters")
    drill_pipe_length = st.number_input("Drill pipe length (ft)", value=float(defaults.get("Drill pipe length (ft)", 5000.0)))
    drill_collar_length = st.number_input("Drill collar length (ft)", value=float(defaults.get("Drill collar length (ft)", 450.0)))
    hwdp_length = st.number_input("HWDP length (ft)", value=float(defaults.get("HWDP length (ft)", 800.0)))
    flow_rate_gpm = st.number_input("Flow Rate (gpm)", value=float(defaults.get("Flow Rate (gpm)", 500.0)))
    drill_pipe_OD = st.number_input("Drill Pipe OD (in)", value=float(defaults.get("Drill Pipe OD (in)", 5.0)))
    drill_pipe_ID = st.number_input("Drill Pipe ID (in)", value=float(defaults.get("Drill Pipe ID (in)", 4.0)))

st.markdown("---")

col_mech, col_env = st.columns(2)

with col_mech:
    st.subheader("Mechanical Parameters")
    rotary_rpm = st.number_input("Rotary Speed (RPM)", value=float(defaults.get("Rotary Speed (RPM)", 150.0)))
    torque_ftlbs = st.number_input("Torque (ft-lbs)", value=float(defaults.get("Torque (ft-lbs)", 15000.0)))
    wob_lbs = st.number_input("Weight on Bit (WOB) (lbs)", value=float(defaults.get("Weight on Bit (WOB) (lbs)", 30000.0)))
    vibration_lvl = st.number_input("Vibration_Level", value=float(defaults.get("Vibration_Level", 2.0)))
    von_mises = st.number_input("Von Mises stress (psi)", value=float(defaults.get("Von Mises stress (psi)", 50000.0)))
    torsional = st.number_input("Torsional stress (psi)", value=float(defaults.get("Torsional stress (psi)", 20000.0)))
    bending = st.number_input("Bending stress (psi)", value=float(defaults.get("Bending stress (psi)", 20000.0)))
    tensile = st.number_input("Tensile stress (psi)", value=float(defaults.get("Tensile stress (psi)", 30000.0)))
    compressive = st.number_input("Compressive stress (psi)", value=float(defaults.get("Compressive stress (psi)", 30000.0)))

with col_env:
    st.subheader("Operational & Environmental")
    bottom_temp_F = st.number_input("Bottom hole temperature (°F)", value=float(defaults.get("Bottom hole temperature (°F)", 200.0)))
    mud_weight_ppg = mud_weight_ppg  # already captured
    dogleg = st.number_input("Dogleg severity (deg/100ft)", value=float(defaults.get("Dogleg severity (deg/100ft)", 2.5)))
    flow_rate_gpm = flow_rate_gpm  # already captured
    inclination = st.number_input("Hole Inclination (deg)", min_value=0.0, max_value=90.0, value=float(defaults.get("inclination", 0.0)))
    depth_ft = st.number_input("Depth (ft)", value=float(defaults.get("Depth (ft)", 5000.0)))

st.markdown("---")

# --------------------
# Build model input dict (match training feature names exactly)
# --------------------
# We will create `full_input` using feature_names from artifacts and fill from the UI where possible,
# otherwise use the dataset median 'defaults' computed earlier or a safe fallback (0).
ui_to_feature_map = {
    # UI name -> dataset feature name (if different). Keep identical where they match.
    "Rotary Speed (RPM)": rotary_rpm,
    "Torque (ft-lbs)": torque_ftlbs,
    "Weight on Bit (WOB) (lbs)": wob_lbs,
    "Vibration_Level": vibration_lvl,
    "Von Mises stress (psi)": von_mises,
    "Torsional stress (psi)": torsional,
    "Bending stress (psi)": bending,
    "Tensile stress (psi)": tensile,
    "Compressive stress (psi)": compressive,
    "Bottom hole temperature (°F)": bottom_temp_F,
    "Mud weight (ppg)": mud_weight_ppg,
    "Drill pipe weight (lbs/ft)": drill_pipe_weight,
    "Drill pipe length (ft)": drill_pipe_length,
    "Drill collar length (ft)": drill_collar_length,
    "HWDP length (ft)": hwdp_length,
    "Flow Rate (gpm)": flow_rate_gpm,
    "Drill Pipe OD (in)": drill_pipe_OD,
    "Drill Pipe ID (in)": drill_pipe_ID,
    "Depth (ft)": depth_ft,
    "Dogleg severity (deg/100ft)": dogleg,
    "inclination": inclination,
    "Yield strength (psi)": yield_strength,
    "Toughness (ft-lbs)": toughness,
    "Hardness (HRC)": hardness,
    # categorical/string fields: use strings matching training data
    "Bit Type": bit_type,
    "Formation Type": formation_type,
    # Risk_Level is a training column in some pipelines — keep Unknown by default
    "Risk_Level": defaults.get("Risk_Level", "Unknown")
}

# When the model expects exact column names, ensure keys match those names exactly.
# Some training pipelines used slightly different names — feature_names contains those exact names.
# We'll prefer values from ui_to_feature_map if the string matches; else use defaults.
def build_full_input(feature_names, ui_map, defaults):
    full = {}
    for feat in feature_names:
        # If the UI provided a value using the same column name, use it:
        if feat in ui_map:
            full[feat] = ui_map[feat]
            continue

        # Sometimes training used slightly different wording, try several likely alternatives:
        alt_found = False
        for alt_key in ui_map:
            # relaxed matching: lower-case compare removing spaces / underscores / parentheses
            n1 = "".join(e for e in feat.lower() if e.isalnum())
            n2 = "".join(e for e in alt_key.lower() if e.isalnum())
            if n1 == n2:
                full[feat] = ui_map[alt_key]
                alt_found = True
                break
        if alt_found:
            continue

        # If default computed from dataset exists, use it
        if feat in defaults:
            full[feat] = defaults[feat]
            continue

        # Fall back: empty string for object-like, 0 for numeric-like
        full[feat] = defaults.get(feat, 0.0)

    return full

# --------------------
# Prediction & display
# --------------------
if st.button("Run Prediction"):
    # build full input with every expected column
    full_input = build_full_input(feature_names, ui_to_feature_map, defaults)

    # Ensure categorical columns are strings (OneHotEncoder expects strings for object type)
    # Identify likely categorical columns by testing defaults values type or by presence of string in defaults
    for c in full_input:
        if isinstance(full_input[c], str):
            full_input[c] = str(full_input[c])
        # If default from dataset looked like string (e.g. bit type) keep as string
        if (c in defaults) and isinstance(defaults[c], str):
            full_input[c] = str(full_input[c])

    # Create DataFrame for preprocessor
    df_input = pd.DataFrame([full_input], columns=feature_names)

    # Safely coerce numeric columns to numeric (preprocessor should handle)
    for col in df_input.columns:
        # preserve string columns that are categorical by checking defaults type
        if col in defaults and isinstance(defaults[col], str):
            df_input[col] = df_input[col].astype(str)
        else:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0.0)

    try:
        X_proc = preprocessor.transform(df_input)
    except Exception as e:
        st.error("Preprocessor.transform failed — feature name mismatch or wrong input types.\n" + str(e))
        st.stop()

    # Model predict probability (0..1)
    prob = float(model.predict(X_proc)[0][0])
    pred = int(prob >= 0.5)

    # Analyze parameter exceedances & recommendations
    exceeded = analyze_parameters(full_input)
    recommendations = generate_recommendations(exceeded, prob, full_input)

    # Explanation: short human readable reasons
    if exceeded:
        explanation_items = [f"{p[0]} {p[3].upper()} (value {p[1]})" for p in exceeded]
        explanation = "; ".join(explanation_items)
    else:
        if prob >= 0.7:
            explanation = "High model predicted risk"
        elif prob >= 0.5:
            explanation = "Moderate model predicted risk"
        else:
            explanation = "Low model predicted risk"

    # Display results
    st.subheader("🔎 Prediction Result")
    st.metric("Prediction", "Failure" if pred == 1 else "No Failure")
    st.metric("Probability", f"{prob*100:.2f} %")
    st.write("**Explanation:**", explanation)

    st.subheader("🛠 Recommendations")
    # show recommendation categories collapsed
    for cat, items in recommendations.items():
        if items:
            with st.expander(cat):
                for it in items:
                    st.write("- " + it)

    # Show exceeded parameter cards small
    if exceeded:
        st.subheader("⚠️ Exceeded Parameters")
        for param, value, threshold, severity, impact in exceeded:
            st.write(f"- **{param}**: {value} (threshold {threshold}) — {severity.upper()} — {impact}")
    # Prepare & download CSV report

    report = pd.DataFrame([full_input])
    report["Prediction"] = "Failure" if pred == 1 else "No Failure"
    report["Probability"] = prob
    report["Explanation"] = explanation
    # Flatten recommendations into single column text for CSV
    rec_text = []
    for cat, items in recommendations.items():
        if items:
            rec_text.append(f"{cat}: " + " | ".join(items))
    report["Recommendations"] = " || ".join(rec_text) if rec_text else ""

    # ensure human-friendly values for bit/formation if numeric coded in training:
    # if model used numeric codes for bit/formation but we passed string values,
    # the data will contain the original strings (ok). If code expects numeric, training's OneHotEncoder will have created columns.
    # For safety, show the display-friendly columns too:
    display_df = report.T.reset_index()
    display_df.columns = ["Parameter", "Value"]

    st.subheader("📄 Report (preview)")
    st.dataframe(display_df)

    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full report (CSV)",
        data=csv,
        file_name=f"drill_failure_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
