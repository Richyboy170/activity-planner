#!/usr/bin/env python3
"""
Test script to verify Random Forest model loads correctly.
This addresses the warning: "No Random Forest results found. Skipping baseline comparison."
"""

import joblib
import numpy as np
from pathlib import Path

def test_rf_loading():
    """Test that the Random Forest model can be loaded successfully."""
    rf_model_path = Path('models/random_forest_baseline.pkl')

    print("=" * 70)
    print("Testing Random Forest Model Loading")
    print("=" * 70)

    # Check if model file exists
    if not rf_model_path.exists():
        print(f"❌ ERROR: Model file not found at {rf_model_path}")
        return False

    print(f"✓ Model file exists: {rf_model_path}")
    print(f"  File size: {rf_model_path.stat().st_size / (1024*1024):.2f} MB")

    # Load the model
    try:
        rf_classifier = joblib.load(rf_model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return False

    # Display model information
    print(f"\nModel Information:")
    print(f"  Type: {type(rf_classifier).__name__}")
    print(f"  Number of estimators: {rf_classifier.n_estimators}")
    print(f"  Max depth: {rf_classifier.max_depth}")
    print(f"  Number of features: {rf_classifier.n_features_in_}")
    print(f"  Number of classes: {rf_classifier.n_classes_}")
    print(f"  Classes: {rf_classifier.classes_}")

    # Test prediction with dummy data
    print(f"\nTesting Prediction Capability:")
    dummy_features = np.random.randn(5, rf_classifier.n_features_in_)
    try:
        predictions = rf_classifier.predict(dummy_features)
        probabilities = rf_classifier.predict_proba(dummy_features)
        print(f"✓ Model can make predictions")
        print(f"  Sample predictions: {predictions}")
        print(f"  Probability shape: {probabilities.shape}")
    except Exception as e:
        print(f"❌ ERROR during prediction: {e}")
        return False

    print("\n" + "=" * 70)
    print("✓ All tests passed! Random Forest model is working correctly.")
    print("=" * 70)

    return True

if __name__ == '__main__':
    success = test_rf_loading()
    exit(0 if success else 1)
