#!/usr/bin/env python3
"""
Save Random Forest Model for Fertilizer Recommendation - Fixed Version
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def save_fertilizer_rf_model_fixed():
    """Train and save Random Forest model for fertilizer recommendation with proper encoding"""
    print("Loading fertilizer dataset...")
    
    # Load the dataset
    fertilizer = pd.read_csv("Fertilizer Prediction.csv")
    
    # Create fertilizer dictionary for encoding
    fert_dict = {
        'Urea': 1,
        'DAP': 2,
        '14-35-14': 3,
        '28-28': 4,
        '17-17-17': 5,
        '20-20': 6,
        '10-26-26': 7,
    }
    
    # Encode target variable
    fertilizer['fert_no'] = fertilizer['Fertilizer Name'].map(fert_dict)
    fertilizer.drop('Fertilizer Name', axis=1, inplace=True)
    
    # Encode categorical columns separately
    soil_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()
    
    fertilizer["Soil Type"] = soil_encoder.fit_transform(fertilizer['Soil Type'])
    fertilizer['Crop Type'] = crop_encoder.fit_transform(fertilizer['Crop Type'])
    
    # Split features and target
    x = fertilizer.drop('fert_no', axis=1)
    y = fertilizer['fert_no']
    
    print(f"Dataset shape: {fertilizer.shape}")
    print(f"Features: {list(x.columns)}")
    print(f"Number of fertilizer classes: {len(y.unique())}")
    print(f"Soil types: {soil_encoder.classes_}")
    print(f"Crop types: {crop_encoder.classes_}")
    
    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
    
    # Scale features
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train_scaled, y_train)
    
    # Evaluate model
    train_accuracy = rf_model.score(x_train_scaled, y_train)
    test_accuracy = rf_model.score(x_test_scaled, y_test)
    
    print(f"Random Forest - Train Accuracy: {train_accuracy:.4f}")
    print(f"Random Forest - Test Accuracy: {test_accuracy:.4f}")
    
    # Save model and scaler
    print("Saving Random Forest model and scaler...")
    pickle.dump(rf_model, open('fertilizer_rf_model.sav', 'wb'))
    pickle.dump(sc, open('fertilizer_rf_scaler.sav', 'wb'))
    
    # Save label encoders separately
    pickle.dump(soil_encoder, open('fertilizer_soil_encoder.sav', 'wb'))
    pickle.dump(crop_encoder, open('fertilizer_crop_encoder.sav', 'wb'))
    
    # Save fertilizer dictionary for reverse mapping
    fert_reverse_dict = {v: k for k, v in fert_dict.items()}
    pickle.dump(fert_reverse_dict, open('fertilizer_dict.sav', 'wb'))
    
    print("Random Forest model and related files saved successfully!")
    
    return rf_model, sc, soil_encoder, crop_encoder, fert_reverse_dict

if __name__ == "__main__":
    save_fertilizer_rf_model_fixed() 