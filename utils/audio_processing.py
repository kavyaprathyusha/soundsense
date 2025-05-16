
import librosa
import numpy as np

def extract_features(file_path, sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        
        features = np.hstack((
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1)
        ))
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None
