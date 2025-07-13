#!/usr/bin/env python3
"""
Save CatBoost Model for Crop Prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import pickle

def save_crop_catboost_model():
    """Train and save CatBoost model for crop prediction"""
    print("Loading crop dataset...")
    
    # Load the dataset
    crop = pd.read_csv("Crop_recommendation.csv")
    
    # Create crop dictionary for encoding
    crop_dict = {
        'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
        'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
        'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15,
        'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
        'kidneybeans': 20, 'chickpea': 21, 'coffee': 22,
    }
    
    # Encode target variable
    crop['crop_no'] = crop['label'].map(crop_dict)
    crop.drop('label', axis=1, inplace=True)
    
    # Split features and target
    x = crop.drop('crop_no', axis=1)
    y = crop['crop_no']
    
    print(f"Dataset shape: {crop.shape}")
    print(f"Features: {list(x.columns)}")
    print(f"Number of crop classes: {len(y.unique())}")
    
    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Scale features
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)
    
    # Train CatBoost model
    print("Training CatBoost model...")
    cat_model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_state=42)
    cat_model.fit(x_train_scaled, y_train)
    
    # Evaluate model
    train_accuracy = cat_model.score(x_train_scaled, y_train)
    test_accuracy = cat_model.score(x_test_scaled, y_test)
    
    print(f"CatBoost - Train Accuracy: {train_accuracy:.4f}")
    print(f"CatBoost - Test Accuracy: {test_accuracy:.4f}")
    
    # Save model and scaler
    print("Saving CatBoost model and scaler...")
    pickle.dump(cat_model, open('crop_catboost_model.sav', 'wb'))
    pickle.dump(sc, open('crop_catboost_scaler.sav', 'wb'))
    
    # Save crop dictionary for reverse mapping
    crop_reverse_dict = {v: k for k, v in crop_dict.items()}
    pickle.dump(crop_reverse_dict, open('crop_dict.sav', 'wb'))
    
    print("CatBoost model and related files saved successfully!")
    
    return cat_model, sc, crop_reverse_dict

if __name__ == "__main__":
    save_crop_catboost_model() 