# Install required packages
# pip install tensorflow keras matplotlib sklearn

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load the EMNIST Dataset (or use the similar MNIST dataset as a placeholder)
# Assuming you have EMNIST dataset or use MNIST dataset here
# EMNIST can be loaded using tensorflow_datasets library
# Here we'll use MNIST for simplicity. Replace it with EMNIST for real handwritten letters.

# Load the EMNIST dataset (if available)
# If using MNIST instead of EMNIST, use the following:
(X_train, y_train), (X_test, y_test) = mnist.load_data()  # Use EMNIST dataset if needed

# 2. Preprocessing: Reshape the data, normalize and one-hot encode the labels
X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
X_test = np.expand_dims(X_test, axis=-1)    # Add channel dimension

# Normalize the images to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode the labels (0-9 for MNIST or A-Z for EMNIST)
y_train = to_categorical(y_train, 10)  # Change 10 to 26 if using EMNIST (A-Z)
y_test = to_categorical(y_test, 10)    # Change 10 to 26 if using EMNIST (A-Z)

# 3. Define the CNN Model
model = models.Sequential()

# First Convolutional Layer: 32 filters, kernel size of 3x3, ReLU activation, input shape (28, 28, 1)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolutional Layer: 64 filters, kernel size of 3x3, ReLU activation
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third Convolutional Layer: 64 filters, kernel size of 3x3, ReLU activation
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the 3D outputs to 1D
model.add(layers.Flatten())

# Fully connected layer with 64 units and ReLU activation
model.add(layers.Dense(64, activation='relu'))

# Output layer with softmax activation (10 or 26 output classes depending on your dataset)
model.add(layers.Dense(10, activation='softmax'))  # Change 10 to 26 for EMNIST (A-Z)

# 4. Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train the Model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# 6. Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# 7. Predict and Visualize Results
# Making predictions on the test set
y_pred = model.predict(X_test)

# Get the predicted class (argmax returns the index of the highest probability)
y_pred_classes = np.argmax(y_pred, axis=1)

# Let's visualize some results
num_images = 10
plt.figure(figsize=(12, 6))
for i in range(num_images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {y_pred_classes[i]}")
    plt.axis('off')
plt.show()

# 8. Classification Report and Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("Classification Report:\n", classification_report(np.argmax(y_test, axis=1), y_pred_classes))
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))  # Change range if using A-Z (0-25)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 9. Save the Model
model.save('handwritten_character_recognition_model.h5')
