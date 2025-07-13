# 🌾 Crop and Fertilizer Recommendation System

This project utilizes machine learning to build two intelligent systems:
- **Crop Recommendation System**: Suggests the most suitable crop to grow based on soil and environmental conditions.
- **Fertilizer Recommendation System**: Recommends the appropriate fertilizer based on soil nutrients and crop requirements.

---

## 📁 Project Structure

| File/Folder | Description |
|-------------|-------------|
| `Crop_Prediction.ipynb` | Jupyter notebook for crop recommendation using input features like N, P, K, temperature, humidity, pH, and rainfall. |
| `Fertilizer_recommendation.ipynb` | Notebook for predicting suitable fertilizers based on nutrient imbalance and crop requirements. |
| `Crop_recommendation.csv` | Dataset used for training the crop prediction model. |
| `Fertilizer Prediction.csv` | Dataset used for training the fertilizer prediction model. |
| `crop_model.sav` | Trained crop prediction ML model (likely a classification model). |
| `crop_scaler.sav` | Scaler used for normalizing crop input features during training. |
| `fertilizer_model.sav` | Trained fertilizer recommendation ML model. |
| `fertilizer_scaler.sav` | Scaler used for fertilizer feature normalization. |
| `*.sav` files (UUIDs) | Additional saved model versions (likely backups or alternate models). |

---

## 🚀 Features

### Crop Recommendation
- Predicts the best crop to grow based on:
  - Nitrogen (N), Phosphorus (P), Potassium (K)
  - Temperature
  - Humidity
  - pH level
  - Rainfall

### Fertilizer Recommendation
- Identifies nutrient deficiencies or excess
- Suggests balanced fertilizer applications for the given crop and soil profile

---

## 🔧 Technologies Used

- Python
- Jupyter Notebook
- Scikit-learn
- Pandas, NumPy, Matplotlib
- Machine Learning models (e.g., Random Forest, Decision Tree)
- `.sav` files used for serialized models and scalers

---

## 📊 Datasets

### 1. `Crop_recommendation.csv`
- Source: [Kaggle](https://www.kaggle.com/datasets)
- Contains crop data with environmental and soil conditions

### 2. `Fertilizer Prediction.csv`
- Contains data mapping soil/crop conditions to suitable fertilizer types

---

## 📌 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run notebooks:
   - `Crop_Prediction.ipynb`
   - `Fertilizer_recommendation.ipynb`

4. Use saved models (`.sav`) and scalers for deployment or integration in a web or mobile app.

---

## 📈 Future Work

- Integrate with a Flask or Django backend
- Build a mobile/web interface for farmers
- Real-time weather and soil data integration using APIs
- Multi-language support for rural accessibility

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.
