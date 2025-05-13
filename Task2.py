# Install necessary packages if you haven't already
# pip install librosa tensorflow scikit-learn matplotlib

import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Load Audio Files and Extract Features (MFCC)
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    mfccs = np.mean(mfccs, axis=1)
    delta = np.mean(delta, axis=1)
    delta2 = np.mean(delta2, axis=1)
    return np.hstack((mfccs, delta, delta2))

# 2. Prepare the Dataset
# Assuming you have a folder with audio files and labels in a format like:
# /dataset/happy/xxx.wav, /dataset/angry/xxx.wav, etc.

dataset_path = 'path_to_your_dataset'  # Folder where your dataset is stored
emotions = ['happy', 'sad', 'angry', 'fear', 'neutral', 'surprise']  # Customize according to your dataset

X = []
y = []

for emotion in emotions:
    emotion_folder = os.path.join(dataset_path, emotion)
    for filename in os.listdir(emotion_folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(emotion_folder, filename)
            features = extract_features(file_path)
            X.append(features)
            y.append(emotion)

X = np.array(X)
y = np.array(y)

# 3. Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build the Deep Learning Model (Using LSTM or CNN)
# Here we use a simple LSTM model for emotion recognition

model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(emotions), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 7. Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# 8. Make Predictions and Show Results
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 9. Plot Training History (Loss and Accuracy)
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
