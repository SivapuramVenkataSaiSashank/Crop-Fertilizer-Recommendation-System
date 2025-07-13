#!/usr/bin/env python3
"""
Enhanced Crop Recommendation System
Using Decision Tree, Random Forest, and Extra Trees algorithms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import machine learning libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle

def load_and_preprocess_data():
    """Load and preprocess the crop recommendation dataset"""
    print("Loading dataset...")
    crop = pd.read_csv("Crop_recommendation.csv")
    
    # Create crop dictionary for encoding
    crop_dict = {
        'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
        'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
        'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15,
        'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
        'kidneybeans': 20, 'chickpea': 21, 'coffee': 22,
    }
    
    # Encode the target variable
    crop['crop_no'] = crop['label'].map(crop_dict)
    crop.drop('label', axis=1, inplace=True)
    
    # Prepare features and target
    x = crop.drop('crop_no', axis=1)
    y = crop['crop_no']
    
    print(f"Dataset shape: {crop.shape}")
    print(f"Features: {list(x.columns)}")
    print(f"Number of crop classes: {len(y.unique())}")
    
    return x, y

def train_models(x, y):
    """Train Decision Tree, Random Forest, and Extra Trees models"""
    print("\nSplitting data...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    print("Scaling features...")
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)
    
    print("Training models...")
    # Initialize models
    dt_model = DecisionTreeClassifier(random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    
    # Train models
    dt_model.fit(x_train_scaled, y_train)
    rf_model.fit(x_train_scaled, y_train)
    et_model.fit(x_train_scaled, y_train)
    
    models = {
        'Decision Tree': dt_model,
        'Random Forest': rf_model,
        'Extra Trees': et_model
    }
    
    return models, sc, x_test_scaled, y_test, x_train_scaled, y_train

def evaluate_models(models, x_test_scaled, y_test, x_train_scaled, y_train):
    """Evaluate all models and compare their performance"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    results = {}
    
    for name, model in models.items():
        # Test predictions
        y_pred = model.predict(x_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Train predictions
        y_pred_train = model.predict(x_train_scaled)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, x_train_scaled, y_train, cv=5)
        
        results[name] = {
            'test_accuracy': test_accuracy,
            'train_accuracy': train_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"\n{name}:")
        print(f"  Test Accuracy:  {test_accuracy:.4f}")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  CV Mean:        {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results

def analyze_feature_importance(models, feature_names):
    """Analyze and visualize feature importance for ensemble models"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature importance for ensemble models
    rf_importance = models['Random Forest'].feature_importances_
    et_importance = models['Extra Trees'].feature_importances_
    
    print("\nRandom Forest Feature Importance:")
    for feature, importance in zip(feature_names, rf_importance):
        print(f"  {feature}: {importance:.4f}")
    
    print("\nExtra Trees Feature Importance:")
    for feature, importance in zip(feature_names, et_importance):
        print(f"  {feature}: {importance:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(feature_names, models['Decision Tree'].feature_importances_)
    plt.title('Decision Tree Feature Importance')
    plt.xticks(rotation=45)
    plt.ylabel('Importance')
    
    plt.subplot(1, 3, 2)
    plt.bar(feature_names, rf_importance)
    plt.title('Random Forest Feature Importance')
    plt.xticks(rotation=45)
    plt.ylabel('Importance')
    
    plt.subplot(1, 3, 3)
    plt.bar(feature_names, et_importance)
    plt.title('Extra Trees Feature Importance')
    plt.xticks(rotation=45)
    plt.ylabel('Importance')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_prediction_function(models, sc):
    """Create an enhanced prediction function"""
    def crop_recommend_enhanced(N, P, K, temperature, humidity, ph, rainfall, model='rf'):
        """
        Enhanced crop recommendation function that can use different models
        
        Parameters:
        - N, P, K: Nitrogen, Phosphorus, Potassium values
        - temperature, humidity, ph, rainfall: Environmental parameters
        - model: 'dt' for Decision Tree, 'rf' for Random Forest, 'et' for Extra Trees
        """
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        transformed_features = sc.transform(features)
        
        # Select model
        if model == 'dt':
            prediction = models['Decision Tree'].predict(transformed_features)
            probabilities = models['Decision Tree'].predict_proba(transformed_features)
            model_name = "Decision Tree"
        elif model == 'rf':
            prediction = models['Random Forest'].predict(transformed_features)
            probabilities = models['Random Forest'].predict_proba(transformed_features)
            model_name = "Random Forest"
        elif model == 'et':
            prediction = models['Extra Trees'].predict(transformed_features)
            probabilities = models['Extra Trees'].predict_proba(transformed_features)
            model_name = "Extra Trees"
        else:
            raise ValueError("Model must be 'dt', 'rf', or 'et'")
        
        confidence = np.max(probabilities) * 100
        
        # Crop dictionary
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 
            11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 
            16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 
            20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }
        
        recommended_crop = crop_dict[prediction[0]]
        
        return f"{recommended_crop} is the best crop to be cultivated (Confidence: {confidence:.2f}% using {model_name})"
    
    return crop_recommend_enhanced

def save_models(models, sc):
    """Save all trained models and scaler"""
    print("\nSaving models...")
    pickle.dump(models['Decision Tree'], open('crop_model_dt.sav', 'wb'))
    pickle.dump(models['Random Forest'], open('crop_model_rf.sav', 'wb'))
    pickle.dump(models['Extra Trees'], open('crop_model_et.sav', 'wb'))
    pickle.dump(sc, open('crop_scaler.sav', 'wb'))
    print("All models and scaler saved successfully!")

def main():
    """Main function to run the enhanced crop prediction system"""
    print("Enhanced Crop Recommendation System")
    print("Using Decision Tree, Random Forest, and Extra Trees")
    print("="*60)
    
    # Load and preprocess data
    x, y = load_and_preprocess_data()
    
    # Train models
    models, sc, x_test_scaled, y_test, x_train_scaled, y_train = train_models(x, y)
    
    # Evaluate models
    results = evaluate_models(models, x_test_scaled, y_test, x_train_scaled, y_train)
    
    # Analyze feature importance
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    analyze_feature_importance(models, feature_names)
    
    # Create prediction function
    predict_crop = create_prediction_function(models, sc)
    
    # Test predictions
    print("\n" + "="*60)
    print("TESTING PREDICTIONS")
    print("="*60)
    
    # Test case 1
    print("\nTest Case 1:")
    N, P, K = 122, 49, 13
    temperature, humidity, ph, rainfall = 29, 42, 6.1, 202
    
    print(f"Parameters: N={N}, P={P}, K={K}, temp={temperature}, humidity={humidity}, ph={ph}, rainfall={rainfall}")
    print(predict_crop(N, P, K, temperature, humidity, ph, rainfall, 'dt'))
    print(predict_crop(N, P, K, temperature, humidity, ph, rainfall, 'rf'))
    print(predict_crop(N, P, K, temperature, humidity, ph, rainfall, 'et'))
    
    # Test case 2
    print("\nTest Case 2:")
    N, P, K = 90, 42, 43
    temperature, humidity, ph, rainfall = 20, 82, 6.1, 202
    
    print(f"Parameters: N={N}, P={P}, K={K}, temp={temperature}, humidity={humidity}, ph={ph}, rainfall={rainfall}")
    print(predict_crop(N, P, K, temperature, humidity, ph, rainfall, 'dt'))
    print(predict_crop(N, P, K, temperature, humidity, ph, rainfall, 'rf'))
    print(predict_crop(N, P, K, temperature, humidity, ph, rainfall, 'et'))
    
    # Save models
    save_models(models, sc)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("The enhanced crop prediction system has been successfully implemented with:")
    print("1. Decision Tree Classifier")
    print("2. Random Forest Classifier (100 estimators)")
    print("3. Extra Trees Classifier (100 estimators)")
    print("\nAll models have been trained, evaluated, and saved.")
    print("Feature importance analysis has been completed and visualized.")
    print("The system is ready for crop recommendations!")

if __name__ == "__main__":
    main() 