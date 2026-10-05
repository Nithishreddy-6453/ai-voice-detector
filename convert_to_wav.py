import os
import librosa
import soundfile as sf
import warnings
import sys

# Suppress standard warnings
warnings.filterwarnings("ignore")

# Define paths
SOURCE_FOLDERS = {"data/ai": "data/ai_clean", "data/real": "data/real_clean"}

def sanitize_dataset():
    print("🛡️ STARTING DATASET SANITIZATION (MP3 -> WAV)...")
    print("   This removes 'Illegal Header' errors by converting to clean WAV format.\n")

    for src_folder, dest_folder in SOURCE_FOLDERS.items():
        if not os.path.exists(src_folder):
            print(f"⚠️ Missing source folder: {src_folder}")
            continue

        # Create destination folder
        os.makedirs(dest_folder, exist_ok=True)
        
        files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.mp3', '.wav'))]
        total = len(files)
        success_count = 0
        
        print(f"📂 Processing {src_folder} ({total} files)...")

        for i, filename in enumerate(files):
            src_path = os.path.join(src_folder, filename)
            # Create a new filename with .wav extension
            new_filename = os.path.splitext(filename)[0] + ".wav"
            dest_path = os.path.join(dest_folder, new_filename)

            # Skip if already exists (resume capability)
            if os.path.exists(dest_path):
                success_count += 1
                continue

            try:
                # 1. Load Audio (Try to force load even if header is slightly bad)
                y, sr = librosa.load(src_path, sr=22050, mono=True)
                
                # 2. Save as clean WAV
                sf.write(dest_path, y, sr)
                success_count += 1
                
            except Exception:
                # If it fails, we just ignore it. No print needed.
                pass

            # Progress Bar
            if i % 100 == 0:
                sys.stdout.write(f"\r   ⏳ Converted {i}/{total} files...")
                sys.stdout.flush()

        print(f"\n   ✅ Finished {src_folder}: {success_count} clean files saved to {dest_folder}.\n")

    print("🎉 SANITIZATION COMPLETE!")
    print("   You now have error-free data in 'data/ai_clean' and 'data/real_clean'.")

if __name__ == "__main__":
    sanitize_dataset()