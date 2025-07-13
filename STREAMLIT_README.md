# 🌾 Crop & Fertilizer Recommendation System - Streamlit Web App

This is a comprehensive web application that provides crop and fertilizer recommendations using machine learning models. The application uses a **CatBoost model** for crop prediction and a **Random Forest model** for fertilizer recommendation.

## 🚀 Features

### 🌱 Crop Recommendation
- **Model**: CatBoost Classifier
- **Accuracy**: 98.64% on test data
- **Features**: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, Rainfall
- **Output**: Recommends the best crop from 22 different crop types

### 💧 Fertilizer Recommendation
- **Model**: Random Forest Classifier
- **Accuracy**: 95.00% on test data
- **Features**: Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorus
- **Output**: Recommends the best fertilizer from 7 different fertilizer types

## 📋 Prerequisites

Make sure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install required packages**:
   ```bash
   pip install -r streamlit_requirements.txt
   ```

3. **Ensure all model files are present**:
   - `fertilizer_rf_model.sav` - Random Forest model for fertilizer prediction
   - `fertilizer_rf_scaler.sav` - Scaler for fertilizer features
   - `fertilizer_label_encoder.sav` - Label encoder for categorical variables
   - `fertilizer_dict.sav` - Fertilizer mapping dictionary
   - `crop_catboost_model.sav` - CatBoost model for crop prediction
   - `crop_catboost_scaler.sav` - Scaler for crop features
   - `crop_dict.sav` - Crop mapping dictionary

## 🚀 Running the Application

1. **Navigate to the project directory**:
   ```bash
   cd path/to/Crop-and-Fertilizer-Recommendation-System-main
   ```

2. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

3. **Open your web browser** and go to the URL shown in the terminal (usually `http://localhost:8501`)

## 📱 How to Use

### Home Page
- Overview of both models
- Quick start guide
- Model performance metrics

### Crop Recommendation
1. Navigate to "🌱 Crop Recommendation" in the sidebar
2. Adjust the sliders for soil parameters (N, P, K, pH)
3. Set environmental conditions (Temperature, Humidity, Rainfall)
4. Click "🌾 Get Crop Recommendation"
5. View the recommended crop with confidence score and probability chart

### Fertilizer Recommendation
1. Navigate to "💧 Fertilizer Recommendation" in the sidebar
2. Set environmental parameters (Temperature, Humidity, Moisture)
3. Choose Soil Type and Crop Type from dropdown menus
4. Adjust soil nutrient levels (Nitrogen, Potassium, Phosphorus)
5. Click "💧 Get Fertilizer Recommendation"
6. View the recommended fertilizer with confidence score and probability chart

### Analysis Page
- Detailed model performance metrics
- Feature importance analysis
- Model comparison
- User recommendations

## 📊 Model Performance

| Model | Training Accuracy | Test Accuracy | Classes |
|-------|------------------|---------------|---------|
| CatBoost (Crop) | 99.89% | 98.64% | 22 |
| Random Forest (Fertilizer) | 100.00% | 95.00% | 7 |

## 🌱 Available Crops (22 types)
Rice, Maize, Jute, Cotton, Coconut, Papaya, Orange, Apple, Muskmelon, Watermelon, Grapes, Mango, Banana, Pomegranate, Lentil, Blackgram, Mungbean, Mothbeans, Pigeonpeas, Kidneybeans, Chickpea, Coffee

## 🧪 Available Fertilizers (7 types)
Urea, DAP, 14-35-14, 28-28, 17-17-17, 20-20, 10-26-26

## 🔧 Troubleshooting

### Common Issues:

1. **Model files not found**:
   - Ensure all `.sav` files are in the same directory as `app.py`
   - Run the model saving scripts first if files are missing

2. **Import errors**:
   - Make sure all required packages are installed
   - Check Python version compatibility

3. **Streamlit not starting**:
   - Verify Streamlit is installed: `pip install streamlit`
   - Check if port 8501 is available

## 📁 File Structure

```
Crop-and-Fertilizer-Recommendation-System-main/
├── app.py                          # Main Streamlit application
├── save_fertilizer_rf_model.py     # Script to save Random Forest model
├── save_crop_catboost_model.py     # Script to save CatBoost model
├── streamlit_requirements.txt      # Python dependencies
├── STREAMLIT_README.md            # This file
├── fertilizer_rf_model.sav         # Saved Random Forest model
├── fertilizer_rf_scaler.sav        # Fertilizer feature scaler
├── fertilizer_label_encoder.sav    # Label encoder
├── fertilizer_dict.sav             # Fertilizer mapping
├── crop_catboost_model.sav         # Saved CatBoost model
├── crop_catboost_scaler.sav        # Crop feature scaler
├── crop_dict.sav                   # Crop mapping
├── Fertilizer Prediction.csv       # Fertilizer dataset
└── Crop_recommendation.csv         # Crop dataset
```

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving the UI/UX
- Enhancing model performance

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset sources for crop and fertilizer recommendations
- Streamlit team for the amazing web framework
- CatBoost and scikit-learn communities for the machine learning libraries 