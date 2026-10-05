import os
import numpy as np
import librosa
import soundfile as sf
import joblib
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import miniaudio  # <--- New library

# --- 1. THE SENTINEL: Crash-Proof Audio Loader ---
def load_audio_robust(file_path, sr=22050):
    """
    Uses 'miniaudio' to decode MP3/WAV directly in memory.
    No FFmpeg installation required.
    """
    try:
        # 1. Read file info
        info = miniaudio.get_file_info(file_path)
        
        # 2. Decode stream to float32 (what librosa expects)
        # This converts MP3/WAV/FLAC automatically
        decoded = miniaudio.decode_file(file_path, sample_rate=sr, nchannels=1, output_format=miniaudio.SampleFormat.FLOAT32)
        
        # 3. Convert to Numpy Array
        y = np.array(decoded.samples, dtype=np.float32)
        
        return y, sr

    except Exception as e:
        print(f"⚠️ Miniaudio failed on {os.path.basename(file_path)}: {e}")
        # Final fallback: Standard Librosa
        try:
            y, sr = librosa.load(file_path, sr=sr)
            return y, sr
        except:
            return None, None

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
MODEL_FILE = "deep_voice_model.pkl"
SCALER_FILE = "deep_scaler.pkl"

# --- 1. THE SENTINEL: Crash-Proof Audio Loader ---
def load_audio_robust(file_path, sr=22050):
    try:
        # Method A: Librosa
        y, sr = librosa.load(file_path, sr=sr)
        return y, sr
    except:
        try:
            # Method B: SoundFile
            y, sr = sf.read(file_path)
            if len(y.shape) > 1: y = y.mean(axis=1)
            return y, sr
        except:
            return None, None

# --- 2. THE INSPECTOR: Feature Extraction ---
def extract_deep_features(file_path):
    y, sr = load_audio_robust(file_path)
    if y is None or len(y) < 500: return None

    try:
        # Extract features (MFCC, Contrast, Chroma, Mel, ZCR)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
        
        return np.hstack([mfcc, contrast, chroma, mel, zcr])
    except:
        return None

# --- 3. THE PREDICTOR (Used by API) ---
def predict_voice(file_path):
    if not os.path.exists(MODEL_FILE):
        return {"error": "Model not trained yet."}
    
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        
        features = extract_deep_features(file_path)
        if features is None: return {"error": "Bad audio file"}
        
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0][prediction] * 100
        
        return {"result": "AI" if prediction == 1 else "Human", "confidence": confidence, "is_ai": bool(prediction==1)}
    except Exception as e:
        return {"error": str(e)}

# --- 4. THE TRAINER (Only runs if called explicitly) ---
def start_training_process(real_path="data/real", ai_path="data/ai"):
    print("🧠 STARTING TRAINING...")
    X, y = [], []
    
    # Load AI
    for f in os.listdir(ai_path):
        if f.endswith(('mp3','wav')):
            feat = extract_deep_features(os.path.join(ai_path, f))
            if feat is not None: X.append(feat); y.append(1)
            
    # Load Real
    for f in os.listdir(real_path):
        if f.endswith(('mp3','wav')):
            feat = extract_deep_features(os.path.join(real_path, f))
            if feat is not None: X.append(feat); y.append(0)

    if not X:
        print("❌ No data found!")
        return

    # Train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    model.fit(X_scaled, y)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("✅ TRAINING COMPLETE.")

# CRITICAL: This ensures training ONLY happens when you run 'python brain.py'
if __name__ == "__main__":
    # Point to the NEW clean WAV folders
    start_training_process(real_path="data/real_clean", ai_path="data/ai_clean")