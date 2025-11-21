import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ============================================
# VIDEO BACKGROUND INJECTION FUNCTION 
# This function uses HTML/CSS to make the video play continuously
# and ensures it stays fixed behind all content.
# ============================================
def set_video_background(video_path):
    st.markdown(
        f"""
        <style>
        /* Target the main Streamlit application container */
        .stApp {{
            background-color: transparent; 
            color: white; /* Set default text color to white for contrast */
        }}
        
        /* Custom CSS for the sidebar to ensure it's readable */
        .st-emotion-cache-vk3ypu {{
            background-color: rgba(0, 0, 0, 0.7); /* Dark semi-transparent background for sidebar */
            padding: 10px;
            border-radius: 10px;
        }}

        /* HTML for the video element */
        #video-background {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%; 
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: -1000; /* Puts the video behind all other content */
            background-size: cover;
            filter: brightness(40%); /* Optional: Darken the video for better text readability */
        }}
        </style>
        
        <video autoplay muted loop id="video-background">
            <source src="{video_path}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )

# ============================================
# STREAMLIT UI SETUP & VIDEO ACTIVATION
# ============================================
st.set_page_config(page_title="AI Exoplanet Transit Detector", layout="wide")

# IMPORTANT: This call runs first and fixes the video in the background
set_video_background("background.mp4") 

st.title("🌌 AI Exoplanet Transit Detector")

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    return keras.models.load_model("exoplanet_transit_model.keras")

try:
    MODEL = load_model()
except Exception as e:
    st.error(f"Could not load the model 'exoplanet_transit_model.keras'. Error: {e}")
    st.stop() 

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

    # Reshape to match CNN input (500, 1)
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

        color = "red" if detected else "lime" 
        icon = "🚨" if detected else "🔭"
        message = "TRANSIT DETECTED!" if detected else "NO TRANSIT"

        st.markdown(f"""
        <div style="border:2px solid {color}; padding:15px; border-radius:10px; text-align:center; background-color: rgba(0, 0, 0, 0.7);">
            <h2 style='color:{color};'>{icon} {message}</h2>
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
        
        # --- Dark Mode & Transparency Adjustments ---
        ax.set_title(f"Prediction: {prob:.2f}%", color='white') 
        ax.set_xlabel("Time Step", color='white')
        ax.set_ylabel("Flux", color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('black') 
        
        # FIX: Set figure background to fully transparent (RGBA 0,0,0,0)
        # This resolves the 'Invalid RGBA argument: transparent' error
        fig.patch.set_facecolor((0, 0, 0, 0))
        
        st.pyplot(fig)
