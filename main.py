from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
import json
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load driving model and encoders at startup (small memory footprint)
model = joblib.load(os.path.join("models", "soundsense_emotion_model.pkl"))
encoders = joblib.load(os.path.join("models", "label_encoders.pkl"))

# Load channel mapping
with open(os.path.join(app.static_folder, "sxm_channels_multi.json")) as f:
    sxm_channels = json.load(f)

# Initialize model references for lazy loading
whisper_model = None
text_tokenizer = None
text_model = None

@app.route("/")
def home():
    return render_template("mode_switch.html")

@app.route("/driving")
def driving():
    return render_template("index.html")

@app.route("/voice")
def voice():
    return render_template("voice_input.html")

@app.route("/predict_emotion", methods=["POST"])
def predict_emotion():
    data = request.get_json()
    try:
        features = {
            col: encoders[col].transform([data[col]])[0]
            for col in ["destination", "duration", "traffic", "speed_zone", "style"]
        }
        df = pd.DataFrame([features])
        pred = model.predict(df)[0]
        emotion = encoders["emotion"].inverse_transform([pred])[0]
        recommendations = sxm_channels.get(emotion, [])
        return jsonify({
            "emotion": emotion,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict_voice", methods=["POST"])
def predict_voice():
    global whisper_model, text_tokenizer, text_model

    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio_file']
    audio_path = "temp_input.wav"
    audio_file.save(audio_path)

    try:
        # Lazy load Whisper model
        if whisper_model is None:
            whisper_model = whisper.load_model("base")

        # Lazy load text emotion model
        if text_tokenizer is None or text_model is None:
            text_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
            text_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

        # Transcription
        transcript = whisper_model.transcribe(audio_path)["text"]
        print(transcript)

        # Emotion prediction from text
        inputs = text_tokenizer(transcript, return_tensors="pt")
        outputs = text_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs).item()
        text_emotion = text_model.config.id2label[pred_idx]

        # Map model emotion to app emotion
        emotion_map = {
            "joy": "Energized",
            "sadness": "Stressed",
            "anger": "Stressed",
            "fear": "Stressed",
            "surprise": "Energized",
            "disgust": "Bored",
            "neutral": "Calm"
        }

        mapped_emotion = emotion_map.get(text_emotion, "Calm")
        recommendations = sxm_channels.get(mapped_emotion, [])

        return jsonify({
            "transcript": transcript.strip(),
            "emotion_text": text_emotion,
            "mapped_emotion": mapped_emotion,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Bind to the port Render provides
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
