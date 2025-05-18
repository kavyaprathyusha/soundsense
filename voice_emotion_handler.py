
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import request, jsonify

# Load models once
whisper_model = whisper.load_model("base")

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
        # Text emotion
        text_emotion = predict_text_emotion(transcript)

        return jsonify({
            "transcript": transcript.strip(),
            "emotion_text": text_emotion
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
