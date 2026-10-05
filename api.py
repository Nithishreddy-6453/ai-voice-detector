import os
import requests
import base64
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from brain import predict_voice

# --- Configuration ---
UPLOAD_FOLDER = "uploads"
API_KEY = "my-secret-hackathon-key"
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Helper: Decode Base64 to Audio (Bulletproof Version) ---
def save_base64_audio(base64_string):
    try:
        # 1. Clean the string (remove newlines, spaces)
        base64_string = re.sub(r'\s+', '', base64_string)

        # 2. Remove header if present (e.g., "data:audio/mp3;base64,")
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        # 3. Fix Padding (The #1 Error Cause)
        missing_padding = len(base64_string) % 4
        if missing_padding:
            base64_string += '=' * (4 - missing_padding)

        # 4. Decode
        audio_data = base64.b64decode(base64_string)
        
        # 5. Save
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], "base64_test.mp3")
        with open(filepath, "wb") as f:
            f.write(audio_data)
            
        return filepath, None # Success

    except Exception as e:
        return None, str(e) # Return the specific error message

@app.route('/', methods=['GET'])
def home():
    return "<h1>✅ AI Voice API is Live</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    # --- 1. SECURITY CHECK ---
    incoming_key = request.headers.get('x-api-key') or request.headers.get('Authorization')
    if incoming_key != API_KEY:
        if incoming_key is not None: 
            return jsonify({"error": "Unauthorized. Wrong API Key."}), 401

    filepath = None
    error_msg = "Unknown error"

    # --- 2. HANDLE JSON INPUT ---
    if request.is_json:
        data = request.get_json()
        print(f"📥 Received Data Keys: {list(data.keys())}") 

        # A. Check for URL
        url = data.get('audioUrl') or data.get('audio_url') or data.get('url')
        if url:
            print(f"🔗 Processing URL...")
            # Download logic inline for simplicity
            try:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], "downloaded.mp3")
                resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                if resp.status_code == 200:
                    with open(filepath, 'wb') as f: f.write(resp.content)
                else:
                    return jsonify({"error": f"URL download failed: {resp.status_code}"}), 400
            except Exception as e:
                return jsonify({"error": f"URL error: {str(e)}"}), 400

        # B. Check for Base64 (audioBase64)
        else:
            b64_str = data.get('audioBase64') or data.get('audio_base64') or data.get('base64')
            if b64_str:
                print(f"🔠 Processing Base64...")
                filepath, error_msg = save_base64_audio(b64_str)
                if not filepath:
                    return jsonify({"error": f"Base64 Decode Failed: {error_msg}"}), 400
            else:
                return jsonify({"error": "Missing 'audioUrl' or 'audioBase64'", "keys_received": list(data.keys())}), 400

    # --- 3. HANDLE FILE UPLOAD ---
    elif 'file' in request.files:
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)

    else:
        return jsonify({"error": "Request type not supported"}), 400

    # --- 4. PREDICT ---
    try:
        result = predict_voice(filepath)
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Analysis Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)