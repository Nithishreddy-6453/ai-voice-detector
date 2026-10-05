import joblib
import numpy as np
import sys
import os

# Import feature extraction
from features import extract_mfcc_features, analyze_vocal_physics

# --- Configuration ---
MODEL_FILENAME = "voice_model.pkl"

def predict_voice(file_path):
    # 1. Load the trained model
    if not os.path.exists(MODEL_FILENAME):
        print(f"❌ Error: Model file '{MODEL_FILENAME}' not found. Run train_model.py first.")
        return

    print(f"Loading model from {MODEL_FILENAME}...")
    model = joblib.load(MODEL_FILENAME)

    # 2. Extract Features (The Math)
    print(f"Analyzing audio: {file_path}")
    features = extract_mfcc_features(file_path)

    if features is None:
        print("❌ Error: Could not extract features from this file.")
        return

    # Reshape for the model (1 sample, many features)
    features = features.reshape(1, -1)

    # 3. Predict
    prediction = model.predict(features)[0]       # 0 or 1
    probabilities = model.predict_proba(features)[0] # e.g. [0.1, 0.9]

    # Get the confidence score of the predicted class
    confidence = probabilities[prediction] * 100

    # 4. Result
    result = "🤖 AI VOICE" if prediction == 1 else "👤 HUMAN VOICE"
    
    print("\n" + "="*30)
    print(f"RESULT: {result}")
    print(f"Confidence: {confidence:.2f}%")
    print("="*30)

    # 5. Explain (Optional Physics)
    # This just shows the Jitter/Shimmer for your own info
    print("\n--- Vocal Physics Analysis ---")
    physics = analyze_vocal_physics(file_path)
    if physics:
        print(f"Jitter (Stability): {physics['jitter']:.5f}")
        print(f"Shimmer (Loudness): {physics['shimmer']:.5f}")
        if prediction == 1 and physics['jitter'] < 0.005:
             print("💡 Insight: Extremely low jitter often indicates synthetic speech.")

if __name__ == "__main__":
    # Check if user provided a filename in the command line
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        # Default test file if none provided
        test_file = "data/ai/ai_sample_1.mp3"  # Change this to test different files!

    predict_voice(test_file)