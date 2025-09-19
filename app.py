import streamlit as st
import pickle
import numpy as np
import time

# ---------------------------
# Load models and scaler
# ---------------------------
try:
    crop_model = pickle.load(open("crop_model.pkl", "rb"))
    fert_model = pickle.load(open("fertilizer_model.pkl", "rb"))
    sc = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'crop_model.pkl', 'fertilizer_model.pkl', and 'scaler.pkl' are in the same directory.")
    st.stop()

# ---------------------------
# Mappings
# ---------------------------
fert_dict = {
    1: 'Urea', 2: 'DAP', 3: '14-35-14', 4: '28-28', 5: '17-17-17',
    6: '20-20', 7: '10-26-26'
}

crop_mapping = {
    0: "Wheat", 1: "Mungbean", 2: "Maize", 3: "Pigeonpeas", 4: "Cotton",
    5: "Blackgram", 6: "Mothbeans", 7: "Lentil", 8: "Jute", 9: "Chickpea",
    10: "Coffee", 11: "Kidneybeans", 12: "Muskmelon", 13: "Rice", 14: "Groundnuts",
    15: "Soybeans", 16: "Potatoes", 17: "Beans", 18: "Peas", 19: "Sugarcane",
    20: "Barley", 21: "Millet", 22: "Sorghum", 23: "Lentils", 24: "Apple",
    25: "Orange", 26: "Mango", 27: "Grapes", 28: "Banana", 29: "Pomegranate",
    30: "Watermelon", 31: "Coconut"
}

soil_dict = {"Sandy": 0, "Clay": 1, "Loamy": 2}

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 3rem;
        border-radius: 1.5rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.3; }
    }
    
    .input-card {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    .input-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.15);
    }
    
    .result-card {
        padding: 2rem;
        border-radius: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
    }
    
    .result-card:hover {
        transform: scale(1.02);
    }
    
    .crop-result {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #1a3c6e;
    }
    
    .fert-result {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #1a3c6e;
    }
    
    .error-result {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        color: #721c24;
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid #e9ecef;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 20px rgba(17, 153, 142, 0.6);
    }
    
    .stNumberInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    
    .input-label {
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.5rem;
    }
    
    .icon-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    .icon-container .icon {
        font-size: 3rem;
        margin: 0 0.5rem;
        animation: float 3s ease-in-out infinite;
    }
    
    .icon-container .icon:nth-child(2) {
        animation-delay: 0.5s;
    }
    
    .icon-container .icon:nth-child(3) {
        animation-delay: 1s;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .section-title .emoji {
        margin-right: 0.5rem;
        font-size: 1.8rem;
    }
    
    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .result-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #11998e;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Main Header
# ---------------------------
st.markdown("""
<div class="main-header">
    <div class="icon-container">
        <div class="icon">🌾</div>
        <div class="icon">🌱</div>
        <div class="icon">🌿</div>
    </div>
    <h1 style="font-size: 3rem; margin-bottom: 0.5rem; font-weight: 700;">AgriVision AI</h1>
    <p style="font-size: 1.4rem; margin-top: 0; opacity: 0.9;">Smart Crop & Fertilizer Recommendation System</p>
    <p style="font-size: 1.1rem; margin-top: 1rem; opacity: 0.8;">Harnessing AI for Precision Agriculture</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Input Sections
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="input-card">
        <div class="section-title">
            <span class="emoji">🌍</span> Soil Properties
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    N = st.number_input("Nitrogen (N)", 0, 200, 50, key="N")
    P = st.number_input("Phosphorus (P)", 0, 200, 50, key="P")
    K = st.number_input("Potassium (K)", 0, 200, 50, key="K")
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, key="pH")
    moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 30.0, key="moisture")
    soil_type = st.selectbox("Soil Type", list(soil_dict.keys()), key="soil_type")

with col2:
    st.markdown("""
    <div class="input-card">
        <div class="section-title">
            <span class="emoji">🌤️</span> Weather Conditions
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    temp = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, key="temp")
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, key="humidity")
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0, key="rainfall")

# ---------------------------
# Recommendation Button
# ---------------------------
st.markdown("<div style='text-align: center; margin: 2.5rem 0;'>", unsafe_allow_html=True)
if st.button("🔍 Get AI Recommendations", key="recommend_btn"):
    # Show loading animation
    st.markdown("""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div style="font-size: 1.2rem; font-weight: 500;">Analyzing your soil and weather data...</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Simulate processing time
    time.sleep(1.5)
    
    # Predict crop
    crop_inputs = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    crop_num = int(crop_model.predict(crop_inputs)[0])
    crop_name = crop_mapping.get(crop_num, "Unknown Crop")

    # Fertilizer model input
    soil_encoded = soil_dict[soil_type]
    fert_inputs = np.array([[temp, humidity, moisture, soil_encoded, crop_num, N, K, P]])
    fert_inputs = sc.transform(fert_inputs)
    fert_num = int(fert_model.predict(fert_inputs)[0])
    fert_name = fert_dict.get(fert_num, "Unknown Fertilizer")

    # Clear loading animation
    st.empty()
    
    # Display results
    st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
    
    # Crop result
    if crop_name == "Unknown Crop":
        st.markdown(f"""
        <div class="result-card error-result">
            <div class="result-title">⚠️ Prediction Error</div>
            <div class="result-value">Unknown Crop</div>
            <p style="font-size: 1.1rem;">The model predicted an unknown value: <code>{crop_num}</code></p>
            <p style="font-size: 1rem;">This value doesn't match any crop in our database. Please check your model training data.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card crop-result">
            <div class="result-title">🌾 Recommended Crop</div>
            <div class="result-value">{crop_name}</div>
            <p style="font-size: 1.1rem;">Based on your soil and weather conditions, this crop is most suitable for optimal growth and yield.</p>
        </div>
        """, unsafe_allow_html=True)

    # Fertilizer result
    if fert_name == "Unknown Fertilizer":
        st.markdown(f"""
        <div class="result-card error-result">
            <div class="result-title">⚠️ Prediction Error</div>
            <div class="result-value">Unknown Fertilizer</div>
            <p style="font-size: 1.1rem;">The model predicted an unknown value: <code>{fert_num}</code></p>
            <p style="font-size: 1rem;">This value doesn't match any fertilizer in our database. Please check your model training data.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card fert-result">
            <div class="result-title">🧪 Recommended Fertilizer</div>
            <div class="result-value">{fert_name}</div>
            <p style="font-size: 1.1rem;">This fertilizer will provide the optimal nutrients for your selected crop and soil conditions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
<div class="footer">
    <p style="margin-bottom: 0.5rem; font-weight: 600;">Developed by Parth Bhegade</p>
    <p style="margin: 0; font-size: 0.9rem;">© 2025 AgriVision AI | AIML Project</p>
    <p style="margin-top: 0.5rem; font-size: 0.8rem; opacity: 0.7;">Empowering Farmers with Artificial Intelligence</p>
</div>
""", unsafe_allow_html=True)
