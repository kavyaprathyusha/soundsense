
import os
import numpy as np
from utils.audio_processing import extract_features
from utils.emotion_classifier import train_classifier

# Simulated labeled dataset
# Replace these with actual paths and labels
# TODO : What is this dataset being used for? - they are used as samples to test the model
dataset = {
    "audio_samples/happy_1.mp3": "happy",
    "audio_samples/happy_2.mp3": "happy",
    "audio_samples/stressed.mp3": "stressed",
    "audio_samples/stressed_2.mp3": "stressed",
    "audio_samples/calm.mp3": "calm",
    "audio_samples/bored.mp3": "bored"
}

X = []
y = []

for file_path, label in dataset.items():
    if os.path.exists(file_path):
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(label)
    else:
        print(f"File not found: {file_path}")

X = np.array(X)
y = np.array(y)

if len(X) > 0:
    model = train_classifier(X, y)
    print("Model trained and saved.")
else:
    print("No training data available.")
