#!/usr/bin/env python3
"""
Test script to verify that both fixed models are working correctly
"""

import pickle
import numpy as np

def test_models_fixed():
    """Test both crop and fertilizer models"""
    print("Testing Fixed Crop & Fertilizer Recommendation Models")
    print("=" * 60)
    
    try:
        # Load crop model
        print("Loading crop model...")
        crop_model = pickle.load(open('crop_catboost_model.sav', 'rb'))
        crop_scaler = pickle.load(open('crop_catboost_scaler.sav', 'rb'))
        crop_dict = pickle.load(open('crop_dict.sav', 'rb'))
        print("✓ Crop model loaded successfully")
        
        # Load fertilizer model
        print("Loading fertilizer model...")
        fertilizer_model = pickle.load(open('fertilizer_rf_model.sav', 'rb'))
        fertilizer_scaler = pickle.load(open('fertilizer_rf_scaler.sav', 'rb'))
        fertilizer_soil_encoder = pickle.load(open('fertilizer_soil_encoder.sav', 'rb'))
        fertilizer_crop_encoder = pickle.load(open('fertilizer_crop_encoder.sav', 'rb'))
        fertilizer_dict = pickle.load(open('fertilizer_dict.sav', 'rb'))
        print("✓ Fertilizer model loaded successfully")
        
        # Test crop prediction
        print("\nTesting crop prediction...")
        test_crop_features = np.array([[90, 42, 43, 20, 82, 6.1, 202]])  # N, P, K, temp, humidity, ph, rainfall
        test_crop_scaled = crop_scaler.transform(test_crop_features)
        crop_prediction = crop_model.predict(test_crop_scaled)
        crop_probabilities = crop_model.predict_proba(test_crop_scaled)[0]
        recommended_crop = crop_dict[int(crop_prediction[0][0])]
        confidence = crop_probabilities[int(crop_prediction[0][0]) - 1] * 100
        
        print(f"Input: N=90, P=42, K=43, temp=20, humidity=82, ph=6.1, rainfall=202")
        print(f"Recommended Crop: {recommended_crop}")
        print(f"Confidence: {confidence:.2f}%")
        
        # Test fertilizer prediction
        print("\nTesting fertilizer prediction...")
        soil_type = "Loamy"
        crop_type = "Wheat"
        soil_encoded = fertilizer_soil_encoder.transform([soil_type])[0]
        crop_encoded = fertilizer_crop_encoder.transform([crop_type])[0]
        
        test_fertilizer_features = np.array([[30, 0.5, 0.5, soil_encoded, crop_encoded, 20, 20, 20]])  # temp, humidity, moisture, soil, crop, N, K, P
        test_fertilizer_scaled = fertilizer_scaler.transform(test_fertilizer_features)
        fertilizer_prediction = fertilizer_model.predict(test_fertilizer_scaled)
        fertilizer_probabilities = fertilizer_model.predict_proba(test_fertilizer_scaled)[0]
        recommended_fertilizer = fertilizer_dict[int(fertilizer_prediction[0])]
        confidence = fertilizer_probabilities[int(fertilizer_prediction[0]) - 1] * 100
        
        print(f"Input: temp=30, humidity=0.5, moisture=0.5, soil={soil_type}, crop={crop_type}, N=20, K=20, P=20")
        print(f"Recommended Fertilizer: {recommended_fertilizer}")
        print(f"Confidence: {confidence:.2f}%")
        
        print("\n" + "=" * 60)
        print("✓ All models tested successfully!")
        print("✓ Streamlit application should work correctly")
        
    except Exception as e:
        print(f"❌ Error testing models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_models_fixed() 