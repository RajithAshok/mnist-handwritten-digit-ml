# MNIST Digit Classifier

This repository contains a beginner-level machine learning project that classifies handwritten digits from the famous MNIST dataset using a simple neural network built with TensorFlow/Keras.

## Project Overview

- Dataset: MNIST handwritten digits (28x28 grayscale images)
- Model: A simple feedforward neural network with one hidden layer
- Training: 5 epochs with Adam optimizer and sparse categorical cross-entropy loss
- Evaluation: Achieves around 97% accuracy on test set
- Visualization includes sample images, model predictions, misclassified examples, and training accuracy/loss plots

---

## Training Output

Below is an example of the training accuracy and loss output from running the model:

<img src="https://github.com/user-attachments/assets/97fba2cf-fe48-4db5-b19c-ef2db27f1c0c" alt="TerminalOutput" width=600 height=600 />


---

## Accuracy and Loss Graph

The graph below shows the model accuracy and loss for both training and validation sets across epochs:

<img src="https://github.com/user-attachments/assets/0edb3180-a919-4453-a7a6-d8bfdcba318f" alt="GraphPlot" width=600 height=600 />

---

## Sample Predictions

Here are some sample test images alongside the model's predicted labels compared with the true labels:

<img src="https://github.com/user-attachments/assets/ccdd168e-b82f-4854-922c-389d311f6789" alt="SamplePredictions" width=600 height=600 />

---

## How to Run

1. Clone this repository.
2. Ensure you have Python 3.x installed.
3. Install required dependencies:
```
pip install tensorflow matplotlib numpy
```
4. Run the `mnist_classifier.ipynb` notebook.

---

Feel free to explore, modify, and improve the model. Pull requests and suggestions are welcome!


