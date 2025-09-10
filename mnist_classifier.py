import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),           # Flatten 2D images to 1D
    tf.keras.layers.Dense(128, activation='relu'),           # Hidden layer with ReLU activation
    tf.keras.layers.Dense(10, activation='softmax')          # Output layer for 10 classes
])

# Compile the model with optimizer, loss, and accuracy metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Train the model for 5 epochs
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training & validation accuracy and loss graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], marker='o')
plt.plot(history.history['val_accuracy'], marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], marker='o')
plt.plot(history.history['val_loss'], marker='o')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

# Visualization - Sample training images
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Predict labels for test set
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Visualization - Predicted vs True labels for first 9 test images
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Pred: {predicted_labels[i]}, True: {y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Visualization - Misclassified images
wrong = np.where(predicted_labels != y_test)[0]
plt.figure(figsize=(8,8))
for i in range(9):
    idx = wrong[i]
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Pred: {predicted_labels[idx]}, True: {y_test[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
