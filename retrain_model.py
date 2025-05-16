import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load the dataset
df = pd.read_csv("realistic_soundsense_training_data.csv")

# Encode each column with its own LabelEncoder
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le  # save the encoder for this column

# Separate features and label
X = df.drop("emotion", axis=1)
y = df["emotion"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ensure the output directory exists
os.makedirs("models", exist_ok=True)

# Save the model and encoders
joblib.dump(model, "models/soundsense_emotion_model.pkl")
joblib.dump(encoders, "models/label_encoders.pkl")

print("✅ Model and encoders saved to 'models/'")
