import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define emotions in RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',                                                                
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Function to extract features from audio
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050)
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # Pad or truncate MFCCs to fixed length
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load RAVDESS dataset
def load_data(data_path):
    X, y = [], []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                # Extract emotion from filename (e.g., 03-01-01-01-01-01-01.wav)
                emotion_code = file.split('-')[2]
                if emotion_code in emotions:
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(emotions[emotion_code])
    return np.array(X), np.array(y)

# Path to RAVDESS dataset (update this to your dataset path)
data_path = 'D:\dataset_speech\dataset'  # Replace with actual path to RAVDESS dataset

# Load and preprocess data
X, y = load_data(data_path)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Reshape X for CNN (add channel dimension)
X = X[..., np.newaxis]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(40, 174, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(emotions), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
model.summary()

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save model
model.save('emotion_speech_model.keras')

# Function to predict emotion from new audio
def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        return "Error processing audio"
    features = features[np.newaxis, ..., np.newaxis]
    prediction = model.predict(features)
    predicted_emotion = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_emotion

# Example usage
if __name__ == "__main__":
    # Test with a sample audio file (replace with your test file path)
    test_file = 'D:\dataset_speech\dataset\audio_speech_actors_01-24'  # Replace with actual file
    predicted_emotion = predict_emotion(test_file)
    print(f"Predicted emotion: {predicted_emotion}")