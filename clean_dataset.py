import os
import librosa
import soundfile as sf
import warnings

# Suppress warnings to keep terminal clean
warnings.filterwarnings("ignore")

# Directories to clean
FOLDERS = ["data/real", "data/ai"]

def is_valid_audio(file_path):
    """
    Tries to load the audio file. Returns False if corrupted.
    """
    try:
        # Check file size (if 0 bytes, it's definitely garbage)
        if os.path.getsize(file_path) < 100:
            return False

        # Try loading with SoundFile (Fast check)
        try:
            sf.read(file_path)
            return True
        except:
            pass

        # Try loading with Librosa (Deep check)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            librosa.load(file_path, sr=None, duration=1.0) # Just read 1 second
        
        return True

    except Exception:
        return False

def clean_data():
    print("🧹 STARTING DATASET CLEANUP...")
    print("   This checks every file to ensure 0 errors during training.\n")
    
    total_deleted = 0
    
    for folder in FOLDERS:
        if not os.path.exists(folder):
            print(f"⚠️ Folder not found: {folder}")
            continue
            
        print(f"📂 Scanning {folder}...")
        files = os.listdir(folder)
        
        for i, filename in enumerate(files):
            file_path = os.path.join(folder, filename)
            
            if not is_valid_audio(file_path):
                print(f"   ❌ Deleting Corrupted File: {filename}")
                try:
                    os.remove(file_path)
                    total_deleted += 1
                except:
                    print("      (Could not delete, verify permissions)")
            
            # Progress update every 500 files
            if i % 500 == 0:
                print(f"      Checked {i}/{len(files)} files...", end="\r")

    print(f"\n\n✅ CLEANUP COMPLETE!")
    print(f"   🗑️  Total Bad Files Removed: {total_deleted}")
    print("   You can now run 'python brain.py' without errors.")

if __name__ == "__main__":
    clean_data()