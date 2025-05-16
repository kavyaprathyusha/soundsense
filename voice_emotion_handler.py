
import os
import whisper
import torchaudio
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from speechbrain.pretrained import SpeakerRecognition
from flask import request, jsonify
import librosa
import numpy as np

# Load models once
whisper_model = whisper.load_model("base")
speechbrain_model = SpeakerRecognition.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="tmp_emotion_model"
)

text_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
text_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

def predict_text_emotion(text):
    inputs = text_tokenizer(text, return_tensors="pt")
    outputs = text_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs).item()
    label = text_model.config.id2label[pred_idx]
    return label

@app.route("/predict_voice", methods=["POST"])
def predict_voice():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio_file']
    audio_path = "temp_input.wav"
    audio_file.save(audio_path)

    try:
        # Transcribe
        transcript = whisper_model.transcribe(audio_path)["text"]

        # Voice emotion (tone)
        signal, sr = torchaudio.load(audio_path)
        signal = signal.mean(dim=0)  # mono
        signal = signal.numpy()
        signal_resampled = librosa.resample(signal, orig_sr=sr, target_sr=16000)
        speechbrain_output = speechbrain_model.classify_file(audio_path)
        tone_emotion = speechbrain_output[3]  # label

        # Text emotion
        text_emotion = predict_text_emotion(transcript)

        return jsonify({
            "transcript": transcript.strip(),
            "emotion_tone": tone_emotion,
            "emotion_text": text_emotion
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
