# 🎧 SoundSense – Emotion-Aware In-Car Audio Assistant

SoundSense is a smart audio experience that detects your mood based on driving behavior and suggests personalized SiriusXM channels.

## 🚀 Features
- Detects emotion from:
    - Destination type
    - Duration
    - Traffic
    - Speed zone
    - Driving style
- Supports 7 emotion categories:
    - Stressed, Calm, Energized, Relaxed, Bored, Game Mode, Romantic
- Recommends multiple SiriusXM channels per emotion
- Simple web UI + Flask backend

## 🛠 How to Run

1. **Install dependencies**  
   pip install -r requirements.txt
2. **Train the model**  
   python retrain_model.py
3. **Start the app**  
   python main.py
4. **Open in browser**  
   Visit [http://localhost:5000](http://localhost:5000)

## 📦 File Structure
- `main.py` – Flask server
- `retrain_model.py` – Model training
- `models/` – Saved model and label encoders
- `templates/index.html` – UI
- `static/style.css` – Styling
- `sxm_channels_multi.json` – Emotion-to-channel mapping

## 📡 Example Use
Input: `Road Trip, 1+, Light, Highway, Calm`  
→ Output: `Emotion: Relaxed`  
→ Channels: Coffee House ☕, Yacht Rock Radio 🛥️

---

Made with ❤️ + 🚗 for SXM Hackathon



Later Steps:

-> Update images for channels 
-> Introduce Voice inputs and classify the audio to emotions
-> 