import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Crop & Fertilizer Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .prediction-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #2E8B57;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all saved models and related files"""
    try:
        # Load fertilizer models
        fertilizer_rf_model = pickle.load(open('fertilizer_rf_model.sav', 'rb'))
        fertilizer_scaler = pickle.load(open('fertilizer_rf_scaler.sav', 'rb'))
        fertilizer_soil_encoder = pickle.load(open('fertilizer_soil_encoder.sav', 'rb'))
        fertilizer_crop_encoder = pickle.load(open('fertilizer_crop_encoder.sav', 'rb'))
        fertilizer_dict = pickle.load(open('fertilizer_dict.sav', 'rb'))
        
        # Load crop models
        crop_catboost_model = pickle.load(open('crop_catboost_model.sav', 'rb'))
        crop_scaler = pickle.load(open('crop_catboost_scaler.sav', 'rb'))
        crop_dict = pickle.load(open('crop_dict.sav', 'rb'))
        
        return {
            'fertilizer_rf_model': fertilizer_rf_model,
            'fertilizer_scaler': fertilizer_scaler,
            'fertilizer_soil_encoder': fertilizer_soil_encoder,
            'fertilizer_crop_encoder': fertilizer_crop_encoder,
            'fertilizer_dict': fertilizer_dict,
            'crop_catboost_model': crop_catboost_model,
            'crop_scaler': crop_scaler,
            'crop_dict': crop_dict
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def predict_fertilizer(models, temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
    """Predict fertilizer recommendation using Random Forest model"""
    try:
        # Encode categorical variables
        soil_encoded = models['fertilizer_soil_encoder'].transform([soil_type])[0]
        crop_encoded = models['fertilizer_crop_encoder'].transform([crop_type])[0]
        
        # Prepare features
        features = np.array([[temperature, humidity, moisture, soil_encoded, crop_encoded, nitrogen, potassium, phosphorous]])
        
        # Scale features
        features_scaled = models['fertilizer_scaler'].transform(features)
        
        # Make prediction
        prediction = int(models['fertilizer_rf_model'].predict(features_scaled)[0])
        probabilities = models['fertilizer_rf_model'].predict_proba(features_scaled)[0]
        
        # Get fertilizer name
        fertilizer_name = models['fertilizer_dict'][prediction]
        confidence = probabilities[prediction - 1] * 100
        
        return fertilizer_name, confidence, probabilities
        
    except Exception as e:
        st.error(f"Error in fertilizer prediction: {str(e)}")
        return None, None, None

def predict_crop(models, N, P, K, temperature, humidity, ph, rainfall):
    """Predict crop recommendation using CatBoost model"""
    try:
        # Prepare features
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Scale features
        features_scaled = models['crop_scaler'].transform(features)
        
        # Make prediction
        prediction = int(models['crop_catboost_model'].predict(features_scaled)[0][0])
        probabilities = models['crop_catboost_model'].predict_proba(features_scaled)[0]
        
        # Get crop name
        crop_name = models['crop_dict'][prediction]
        confidence = probabilities[prediction - 1] * 100
        
        return crop_name, confidence, probabilities
        
    except Exception as e:
        st.error(f"Error in crop prediction: {str(e)}")
        return None, None, None

def create_fertilizer_chart(probabilities, fertilizer_dict):
    """Create a bar chart for fertilizer probabilities"""
    fertilizer_names = list(fertilizer_dict.values())
    
    fig = px.bar(
        x=fertilizer_names,
        y=probabilities * 100,
        title="Fertilizer Recommendation Probabilities",
        labels={'x': 'Fertilizer', 'y': 'Probability (%)'},
        color=probabilities * 100,
        color_continuous_scale='Greens'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False
    )
    
    return fig

def create_crop_chart(probabilities, crop_dict):
    """Create a bar chart for crop probabilities"""
    crop_names = list(crop_dict.values())
    
    # Get top 10 crops by probability
    top_indices = np.argsort(probabilities)[-10:][::-1]
    top_crops = [crop_names[i] for i in top_indices]
    top_probs = [probabilities[i] * 100 for i in top_indices]
    
    fig = px.bar(
        x=top_crops,
        y=top_probs,
        title="Top 10 Crop Recommendations",
        labels={'x': 'Crop', 'y': 'Probability (%)'},
        color=top_probs,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    # Load models
    models = load_models()
    
    if models is None:
        st.error("Failed to load models. Please ensure all model files are present.")
        return
    
    # Main header
    st.markdown('<h1 class="main-header">🌾 Crop & Fertilizer Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🌱 Crop Recommendation", "💧 Fertilizer Recommendation", "📊 Analysis"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🌱 Crop Recommendation":
        show_crop_recommendation(models)
    elif page == "💧 Fertilizer Recommendation":
        show_fertilizer_recommendation(models)
    elif page == "📊 Analysis":
        show_analysis_page()

def show_home_page():
    """Display the home page"""
    st.markdown('<h2 class="sub-header">Welcome to the Crop & Fertilizer Recommendation System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌱 Crop Recommendation
        Our advanced CatBoost model analyzes soil and environmental conditions to recommend the best crop for your land.
        
        **Features analyzed:**
        - Nitrogen (N) content
        - Phosphorus (P) content  
        - Potassium (K) content
        - Temperature
        - Humidity
        - pH level
        - Rainfall
        """)
        
        st.info("**Model Accuracy:** 99.32% on test data")
    
    with col2:
        st.markdown("""
        ### 💧 Fertilizer Recommendation
        Our Random Forest model suggests the optimal fertilizer based on soil conditions and crop type.
        
        **Features analyzed:**
        - Temperature
        - Humidity
        - Moisture
        - Soil Type
        - Crop Type
        - Nitrogen content
        - Potassium content
        - Phosphorus content
        """)
        
        st.info("**Model Accuracy:** 99.10% on test data")
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown('<h3 class="sub-header">🚀 Quick Start Guide</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **For Crop Recommendation:** Navigate to the Crop Recommendation page and input your soil parameters
    2. **For Fertilizer Recommendation:** Navigate to the Fertilizer Recommendation page and provide soil and crop information
    3. **View Analysis:** Check the Analysis page for detailed insights and model performance metrics
    """)
    
    # Model performance metrics
    st.markdown('<h3 class="sub-header">📈 Model Performance</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Crop Model Accuracy", "99.32%")
    with col2:
        st.metric("Fertilizer Model Accuracy", "99.10%")
    with col3:
        st.metric("Crop Classes", "22")
    with col4:
        st.metric("Fertilizer Classes", "7")

def show_crop_recommendation(models):
    """Display crop recommendation page"""
    st.markdown('<h2 class="sub-header">🌱 Crop Recommendation</h2>', unsafe_allow_html=True)
    
    st.markdown("Enter the soil and environmental parameters to get crop recommendations:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Soil Parameters")
        N = st.slider("Nitrogen (N) - kg/ha", 0, 140, 50)
        P = st.slider("Phosphorus (P) - kg/ha", 5, 145, 50)
        K = st.slider("Potassium (K) - kg/ha", 5, 205, 50)
        ph = st.slider("pH", 3.5, 10.0, 6.5)
    
    with col2:
        st.subheader("Environmental Parameters")
        temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0)
        humidity = st.slider("Humidity (%)", 14.0, 100.0, 70.0)
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)
    
    # Prediction button
    if st.button("🌾 Get Crop Recommendation", type="primary"):
        with st.spinner("Analyzing soil conditions..."):
            crop_name, confidence, probabilities = predict_crop(models, N, P, K, temperature, humidity, ph, rainfall)
            
            if crop_name:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### 🌾 Recommended Crop: **{crop_name.title()}**")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display probability chart
                st.plotly_chart(create_crop_chart(probabilities, models['crop_dict']), use_container_width=True)
                
                # Additional information
                st.markdown("### 📋 Input Parameters Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Nitrogen (N):** {N} kg/ha")
                    st.write(f"**Phosphorus (P):** {P} kg/ha")
                    st.write(f"**Potassium (K):** {K} kg/ha")
                    st.write(f"**pH:** {ph}")
                with col2:
                    st.write(f"**Temperature:** {temperature}°C")
                    st.write(f"**Humidity:** {humidity}%")
                    st.write(f"**Rainfall:** {rainfall} mm")

def show_fertilizer_recommendation(models):
    """Display fertilizer recommendation page"""
    st.markdown('<h2 class="sub-header">💧 Fertilizer Recommendation</h2>', unsafe_allow_html=True)
    
    st.markdown("Enter the soil and crop parameters to get fertilizer recommendations:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Environmental Parameters")
        temperature = st.slider("Temperature (°C)", 0, 100, 30)
        humidity = st.slider("Humidity", 0.0, 1.0, 0.5)
        moisture = st.slider("Moisture", 0.0, 1.0, 0.5)
        
        st.subheader("Soil Parameters")
        nitrogen = st.slider("Nitrogen (kg/ha)", 0, 50, 20)
        potassium = st.slider("Potassium (kg/ha)", 0, 50, 20)
        phosphorous = st.slider("Phosphorous (kg/ha)", 0, 50, 20)
    
    with col2:
        st.subheader("Crop & Soil Type")
        soil_types = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
        soil_type = st.selectbox("Soil Type", soil_types)
        
        crop_types = ["Wheat", "Maize", "Cotton", "Tobacco", "Paddy", "Barley", "Millets", "Oil seeds", "Pulses", "Ground Nuts", "Sugarcane"]
        crop_type = st.selectbox("Crop Type", crop_types)
    
    # Prediction button
    if st.button("💧 Get Fertilizer Recommendation", type="primary"):
        with st.spinner("Analyzing soil and crop conditions..."):
            fertilizer_name, confidence, probabilities = predict_fertilizer(
                models, temperature, humidity, moisture, soil_type, crop_type, 
                nitrogen, potassium, phosphorous
            )
            
            if fertilizer_name:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### 💧 Recommended Fertilizer: **{fertilizer_name}**")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display probability chart
                st.plotly_chart(create_fertilizer_chart(probabilities, models['fertilizer_dict']), use_container_width=True)
                
                # Additional information
                st.markdown("### 📋 Input Parameters Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Temperature:** {temperature}°C")
                    st.write(f"**Humidity:** {humidity}")
                    st.write(f"**Moisture:** {moisture}")
                    st.write(f"**Soil Type:** {soil_type}")
                with col2:
                    st.write(f"**Crop Type:** {crop_type}")
                    st.write(f"**Nitrogen:** {nitrogen} kg/ha")
                    st.write(f"**Potassium:** {potassium} kg/ha")
                    st.write(f"**Phosphorous:** {phosphorous} kg/ha")

def show_analysis_page():
    """Display analysis and insights page"""
    st.markdown('<h2 class="sub-header">📊 Analysis & Insights</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌱 Crop Model Performance")
        st.markdown("""
        **CatBoost Model Metrics:**
        - Training Accuracy: 99.89%
        - Test Accuracy: 99.32%
        - Number of Classes: 22
        - Features: 7 (N, P, K, temperature, humidity, pH, rainfall)
        """)
        
        # Feature importance for crop model
        st.markdown("### 📈 Crop Model Feature Importance")
        feature_importance_data = {
            'Feature': ['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall'],
            'Importance': [0.1089, 0.1436, 0.1812, 0.0757, 0.2113, 0.0523, 0.2270]
        }
        
        fig = px.bar(
            feature_importance_data,
            x='Feature',
            y='Importance',
            title="Crop Model Feature Importance",
            color='Importance',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💧 Fertilizer Model Performance")
        st.markdown("""
        **Random Forest Model Metrics:**
        - Training Accuracy: 100.00%
        - Test Accuracy: 99.10%
        - Number of Classes: 7
        - Features: 8 (Temperature, Humidity, Moisture, Soil Type, Crop Type, N, K, P)
        """)
        
        # Available fertilizers
        st.markdown("### 🧪 Available Fertilizers")
        fertilizers = ["Urea", "DAP", "14-35-14", "28-28", "17-17-17", "20-20", "10-26-26"]
        for i, fert in enumerate(fertilizers, 1):
            st.write(f"{i}. {fert}")
    
    st.markdown("---")
    
    # Model comparison
    st.markdown("### 🔍 Model Comparison")
    
    comparison_data = {
        'Model': ['Random Forest (Crop)', 'CatBoost (Fertilizer)'],
        'Training Accuracy': [99.89, 100.00],
        'Test Accuracy': [99.3, 99.1],
        'Number of Classes': [22, 7]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Recommendations for users
    st.markdown("### 💡 Recommendations for Users")
    st.markdown("""
    1. **For Best Results:** Use both models together - first get crop recommendation, then use that crop type for fertilizer recommendation
    2. **Parameter Accuracy:** Ensure soil test results are accurate for better predictions
    3. **Seasonal Considerations:** Consider seasonal variations in environmental parameters
    4. **Local Adaptation:** Adjust recommendations based on local agricultural practices and conditions
    """)

if __name__ == "__main__":
    main() 