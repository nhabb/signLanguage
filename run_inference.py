"""
Run inference on landmarks data using pre-trained models
This script uses only built-in Python libraries for basic functionality
"""

import pickle
import csv
import os
import numpy as np

def load_model_and_predict():
    """Load the trained model and make predictions on landmarks data"""
    
    # Paths
    model_path = "output/models/random_forest.pkl"
    scaler_path = "output/models/scaler.pkl"
    label_encoder_path = "output/models/label_encoder.pkl"
    data_path = "landmarks/landmarks_fixed.csv"
    
    print("Loading trained model and artifacts...")
    
    # Load the trained artifacts
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print("✓ Model and artifacts loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"Model type: {type(model).__name__}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Load and process the landmarks data
    print("\nProcessing landmarks data...")
    
    predictions = []
    sample_count = 0
    
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 3:  # Skip empty or malformed rows
                continue
                
            # Extract features (exclude label column at index 0)
            features = []
            for i in range(1, len(row)):  # Start from index 1 to skip label only
                try:
                    features.append(float(row[i]))
                except ValueError:
                    features.append(0.0)
            
            # Convert to numpy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            # Apply scaling
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            predicted_label = label_encoder.inverse_transform([prediction])[0]
            
            predictions.append({
                'file': 'sample_' + str(sample_count),
                'predicted_label': predicted_label,
                'confidence': max(model.predict_proba(features_scaled)[0])
            })
            
            sample_count += 1
            if sample_count <= 10:  # Show first 10 predictions
                print(f"Sample {sample_count}: Predicted: {predicted_label} (confidence: {max(model.predict_proba(features_scaled)[0]):.3f})")
    
    print(f"\n✓ Processed {sample_count} samples")
    
    # Show prediction distribution
    from collections import Counter
    pred_counts = Counter(p['predicted_label'] for p in predictions)
    print(f"\nPrediction distribution:")
    for label, count in pred_counts.most_common(10):
        percentage = (count / len(predictions)) * 100
        print(f"  {label}: {count} samples ({percentage:.1f}%)")
    
    # Save predictions
    output_path = "output/predictions.csv"
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'predicted_label', 'confidence'])
        for pred in predictions:
            writer.writerow([pred['file'], pred['predicted_label'], pred['confidence']])
    
    print(f"\n✓ Predictions saved to {output_path}")
    print(f"Total predictions made: {len(predictions)}")

if __name__ == "__main__":
    load_model_and_predict()
