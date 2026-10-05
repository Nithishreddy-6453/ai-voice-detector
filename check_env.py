import sys
import numpy
import librosa
import parselmouth
import sklearn
import fastapi
import uvicorn

print("\n" + "="*30)
print("✅ SUCCESS: All Audio AI tools are ready!")
print("="*30)
print(f"Python Version:  {sys.version.split()[0]}")
print(f"Numpy:           {numpy.__version__}")
print(f"Librosa:         {librosa.__version__}")
print(f"Parselmouth:     {parselmouth.__version__}")
print(f"Scikit-learn:    {sklearn.__version__}")
print(f"FastAPI:         {fastapi.__version__}")
print("="*30 + "\n")