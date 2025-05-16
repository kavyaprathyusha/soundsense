
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

MODEL_PATH = os.path.join("models", "rf_emotion_model.pkl")

def train_classifier(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

def load_classifier():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        print("No trained model found.")
        return None

def predict_emotion(model, features):
    if model and features is not None:
        prediction = model.predict([features])
        return prediction[0]
    return "unknown"
