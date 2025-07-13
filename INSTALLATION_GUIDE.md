# Installation Guide for Enhanced Crop Prediction System

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## 🚀 Quick Installation

### Method 1: Using requirements.txt (Recommended)

```bash
# Clone or download the project files
# Navigate to the project directory
cd Crop-and-Fertilizer-Recommendation-System-main

# Install all required libraries
pip install -r requirements.txt
```

### Method 2: Manual Installation

If you prefer to install libraries individually:

```bash
# Core data science libraries
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install matplotlib>=3.4.0
pip install seaborn>=0.11.0

# Machine learning libraries
pip install scikit-learn>=1.0.0

# Jupyter notebook (optional, for interactive development)
pip install jupyter>=1.0.0
pip install notebook>=6.4.0
```

## 📦 Required Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.21.0 | Numerical computing |
| `pandas` | ≥1.3.0 | Data manipulation and analysis |
| `matplotlib` | ≥3.4.0 | Plotting and visualization |
| `seaborn` | ≥0.11.0 | Statistical data visualization |
| `scikit-learn` | ≥1.0.0 | Machine learning algorithms |
| `jupyter` | ≥1.0.0 | Interactive development environment |
| `notebook` | ≥6.4.0 | Jupyter notebook interface |

## 🔧 Verification

To verify that all libraries are installed correctly:

```bash
# Run the enhanced crop prediction script
python crop_prediction_enhanced.py
```

You should see output similar to:
```
Enhanced Crop Recommendation System
Using Decision Tree, Random Forest, and Extra Trees
============================================================
Loading dataset...
Dataset shape: (2200, 8)
...
```

## 🐍 Using Conda (Alternative)

If you prefer using Conda:

```bash
# Create a new conda environment
conda create -n crop_prediction python=3.9

# Activate the environment
conda activate crop_prediction

# Install libraries
conda install numpy pandas matplotlib seaborn scikit-learn jupyter notebook

# Or install from requirements.txt
pip install -r requirements.txt
```

## 📁 Project Structure

After installation, your project should contain:

```
Crop-and-Fertilizer-Recommendation-System-main/
├── Crop_recommendation.csv          # Dataset
├── crop_prediction_enhanced.py      # Enhanced Python script
├── Crop_Prediction_Enhanced.ipynb   # Enhanced Jupyter notebook
├── requirements.txt                 # Dependencies
├── INSTALLATION_GUIDE.md           # This file
├── crop_model_dt.sav               # Decision Tree model (generated)
├── crop_model_rf.sav               # Random Forest model (generated)
├── crop_model_et.sav               # Extra Trees model (generated)
├── crop_scaler.sav                 # Scaler (generated)
└── feature_importance.png          # Feature importance plot (generated)
```

## 🚨 Troubleshooting

### Common Issues:

1. **Permission Error**: Use `pip install --user -r requirements.txt`

2. **Version Conflicts**: Create a virtual environment:
   ```bash
   python -m venv crop_env
   source crop_env/bin/activate  # On Windows: crop_env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Missing Dependencies**: Install system dependencies:
   - Windows: Visual C++ Build Tools
   - Linux: `sudo apt-get install python3-dev`
   - macOS: Xcode Command Line Tools

4. **Jupyter Issues**: If Jupyter doesn't start:
   ```bash
   jupyter notebook --generate-config
   jupyter notebook
   ```

## ✅ Success Indicators

After successful installation, you should be able to:

1. ✅ Run `python crop_prediction_enhanced.py` without errors
2. ✅ See model training and evaluation output
3. ✅ Generate feature importance visualizations
4. ✅ Make crop predictions with confidence scores
5. ✅ Save and load trained models

## 🎯 Next Steps

Once installation is complete:

1. Run the enhanced script: `python crop_prediction_enhanced.py`
2. Explore the Jupyter notebook: `jupyter notebook Crop_Prediction_Enhanced.ipynb`
3. Try different input parameters for crop recommendations
4. Analyze feature importance and model performance

## 📞 Support

If you encounter any issues:

1. Check that all dependencies are installed: `pip list`
2. Verify Python version: `python --version`
3. Ensure you're in the correct directory with the dataset file
4. Check the console output for specific error messages

---

**Happy Crop Predicting! 🌱** 