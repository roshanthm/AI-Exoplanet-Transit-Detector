import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    return keras.models.load_model("exoplanet_transit_model.keras")

MODEL = load_model()
NUM_POINTS = 500

# ============================================
# LIGHT CURVE GENERATOR
# ============================================
def generate_light_curve(num_points, has_transit, transit_depth, transit_duration, transit_noise):
    curve = np.ones(num_points)

    # Base noise
    curve += np.random.normal(0, transit_noise * 0.5, num_points)

    if has_transit:
        transit_center = np.random.randint(num_points // 3, 2 * num_points // 3)

        for i in range(num_points):
            d = abs(i - transit_center)

            # Flat bottom
            if d < transit_duration // 2:
                curve[i] -= transit_depth

            # Slope ingress/egress
            elif d < transit_duration // 2 + 10:
                slope = 1 - ((d - transit_duration // 2) / 10)
                curve[i] -= transit_depth * slope

    # Extra noise
    curve += np.random.normal(0, transit_noise, num_points)
    return curve

def normalize_curve(curve):
    return (curve - np.mean(curve)) / (np.std(curve) + 1e-8)

# ============================================
# STREAMLIT UI
# ============================================
st.set_page_config(page_title="AI Exoplanet Transit Detector", layout="wide")

st.title("🌌 AI Exoplanet Transit Detector")

# SIDEBAR — USER CONTROLS
st.sidebar.header("Light Curve Settings")

has_transit = st.sidebar.checkbox("Transit Present?", True)
depth = st.sidebar.slider("Transit Depth", 0.01, 0.15, 0.05, 0.01)
duration = st.sidebar.slider("Transit Duration", 30, 150, 80, 10)
noise = st.sidebar.slider("Noise Level", 0.01, 0.10, 0.03, 0.01)

run_button = st.sidebar.button("Generate & Predict", type="primary")

# AUTO RUN first time
if run_button or "first_run" not in st.session_state:
    st.session_state["first_run"] = True

    # Generate the curve
    curve = generate_light_curve(NUM_POINTS, has_transit, depth, duration, noise)

    # Normalize
    norm_curve = normalize_curve(curve)

    # FIX: reshape to match CNN input (500, 1)
    norm_curve = norm_curve.reshape(NUM_POINTS, 1)
    model_input = norm_curve.reshape(1, NUM_POINTS, 1)

    # Predict
    prob = MODEL.predict(model_input, verbose=0)[0][0] * 100
    detected = prob > 50

    # ============================================
    # OUTPUT COLUMNS
    # ============================================
    col1, col2 = st.columns([3, 1])

    # RESULT PANEL
    with col2:
        st.subheader("Prediction Result")

        color = "red" if detected else "green"
        icon = "🚨" if detected else "🔭"
        message = "TRANSIT DETECTED!" if detected else "NO TRANSIT"

        st.markdown(f"""
        <div style="border:2px solid {color}; padding:15px; border-radius:10px; text-align:center;">
            <h2>{icon} {message}</h2>
            <p style='font-size:20px;'>Probability:</p>
            <p style='font-size:36px; font-weight:bold; color:{color};'>{prob:.2f}%</p>
            <small>Actual transit: {has_transit}</small>
        </div>
        """, unsafe_allow_html=True)

        st.write("### Parameters")
        st.write(f"**Depth**: {depth}")
        st.write(f"**Duration**: {duration}")
        st.write(f"**Noise**: {noise}")

    # PLOT PANEL
    with col1:
        st.subheader("Generated Light Curve")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(curve, color="cyan")
        ax.set_title(f"Prediction: {prob:.2f}%")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Flux")

        st.pyplot(fig)
