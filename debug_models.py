#!/usr/bin/env python3
"""
Debug script to understand model structure
"""

import pickle
import numpy as np

def debug_models():
    """Debug the saved models and dictionaries"""
    print("Debugging Crop & Fertilizer Models")
    print("=" * 50)
    
    try:
        # Load crop model
        print("Loading crop model...")
        crop_model = pickle.load(open('crop_catboost_model.sav', 'rb'))
        crop_scaler = pickle.load(open('crop_catboost_scaler.sav', 'rb'))
        crop_dict = pickle.load(open('crop_dict.sav', 'rb'))
        print("✓ Crop model loaded successfully")
        print(f"Crop dict type: {type(crop_dict)}")
        print(f"Crop dict keys: {list(crop_dict.keys())[:5]}...")
        print(f"Crop dict values: {list(crop_dict.values())[:5]}...")
        
        # Load fertilizer model
        print("\nLoading fertilizer model...")
        fertilizer_model = pickle.load(open('fertilizer_rf_model.sav', 'rb'))
        fertilizer_scaler = pickle.load(open('fertilizer_rf_scaler.sav', 'rb'))
        fertilizer_label_encoder = pickle.load(open('fertilizer_label_encoder.sav', 'rb'))
        fertilizer_dict = pickle.load(open('fertilizer_dict.sav', 'rb'))
        print("✓ Fertilizer model loaded successfully")
        print(f"Fertilizer dict type: {type(fertilizer_dict)}")
        print(f"Fertilizer dict keys: {list(fertilizer_dict.keys())}")
        print(f"Fertilizer dict values: {list(fertilizer_dict.values())}")
        
        # Test crop prediction
        print("\nTesting crop prediction...")
        test_crop_features = np.array([[90, 42, 43, 20, 82, 6.1, 202]])
        test_crop_scaled = crop_scaler.transform(test_crop_features)
        crop_prediction = crop_model.predict(test_crop_scaled)
        print(f"Crop prediction type: {type(crop_prediction)}")
        print(f"Crop prediction shape: {crop_prediction.shape}")
        print(f"Crop prediction value: {crop_prediction}")
        print(f"Crop prediction [0]: {crop_prediction[0]}")
        print(f"Crop prediction [0] type: {type(crop_prediction[0])}")
        
        crop_prediction_int = int(crop_prediction[0])
        print(f"Crop prediction as int: {crop_prediction_int}")
        print(f"Crop prediction as int type: {type(crop_prediction_int)}")
        
        if crop_prediction_int in crop_dict:
            recommended_crop = crop_dict[crop_prediction_int]
            print(f"Recommended Crop: {recommended_crop}")
        else:
            print(f"❌ Crop prediction {crop_prediction_int} not found in crop_dict")
            print(f"Available keys: {list(crop_dict.keys())}")
        
        # Test fertilizer prediction
        print("\nTesting fertilizer prediction...")
        soil_type = "Loamy"
        crop_type = "Wheat"
        
        print(f"Available soil types in encoder: {fertilizer_label_encoder.classes_}")
        print(f"Available crop types in encoder: {fertilizer_label_encoder.classes_}")
        
        try:
            soil_encoded = fertilizer_label_encoder.transform([soil_type])[0]
            crop_encoded = fertilizer_label_encoder.transform([crop_type])[0]
            print(f"Soil '{soil_type}' encoded as: {soil_encoded}")
            print(f"Crop '{crop_type}' encoded as: {crop_encoded}")
        except Exception as e:
            print(f"❌ Error encoding: {str(e)}")
            return
        
        test_fertilizer_features = np.array([[30, 0.5, 0.5, soil_encoded, crop_encoded, 20, 20, 20]])
        test_fertilizer_scaled = fertilizer_scaler.transform(test_fertilizer_features)
        fertilizer_prediction = fertilizer_model.predict(test_fertilizer_scaled)
        print(f"Fertilizer prediction type: {type(fertilizer_prediction)}")
        print(f"Fertilizer prediction shape: {fertilizer_prediction.shape}")
        print(f"Fertilizer prediction value: {fertilizer_prediction}")
        
        fertilizer_prediction_int = int(fertilizer_prediction[0])
        print(f"Fertilizer prediction as int: {fertilizer_prediction_int}")
        
        if fertilizer_prediction_int in fertilizer_dict:
            recommended_fertilizer = fertilizer_dict[fertilizer_prediction_int]
            print(f"Recommended Fertilizer: {recommended_fertilizer}")
        else:
            print(f"❌ Fertilizer prediction {fertilizer_prediction_int} not found in fertilizer_dict")
            print(f"Available keys: {list(fertilizer_dict.keys())}")
        
    except Exception as e:
        print(f"❌ Error debugging models: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_models() 