## 🗂 Project Title

📜 Handwriting Text Recognition & Generation (RNN)

## 🚀 Project Overview

This project focuses on generating realistic handwritten-style text which implements a Deep Learning pipeline to bridge the gap between static image data and sequential character modeling. Using the EMNIST Balanced dataset, the system learns to recognize and generate handwriting by treating character images as time-series sequences of pixels.

## 🗂 Key Features

Handwriting image preprocessing

Sequence modeling

Generative text synthesis

AI-based handwriting simulation

## 🧑‍💻Models Used

RNN

## 🧠 Model Architecture

We utilize a Stacked LSTM (Long Short-Term Memory) architecture designed to capture the spatial dependencies of pen strokes.

Input Layer: 28-pixel features (mapping one row of an image per time-step).

Recurrent Layers: 2-Layer LSTM with 128 hidden units.

Regularization: Dropout (0.2) to ensure generalization across different handwriting styles.

Output Layer: Dense layer with Softmax activation for 47-class character classification.

## 📊 Dataset & Preprocessing

Source: EMNIST Balanced (Extended MNIST).

Normalization: Pixel values scaled to [0, 1] for faster gradient convergence.

Orientation Fix: 90-degree rotation and horizontal flip applied via Transpose (.T) to correct raw dataset storage formats.

## ⚙️ Training & Evaluation Strategy

Optimizer: AdamW (Weight Decay) for superior generalization.

Loss Function: Cross-Entropy Loss.

Metric Focus: F1-Score (to balance precision and recall across potentially imbalanced character classes).

Visual Debugging: Confusion Matrix Heatmaps to identify "Character Overlap" (e.g., confusing '5' with 'S').

## 🛠 Technologies Used

Core: Python, PyTorch (Deep Learning)

Data: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Metrics: Scikit-Learn

TensorFlow / Keras

## Dataset

[Dataset: Handwriting](https://www.kaggle.com/datasets/crawford/emnist?utm_source=chatgpt.com)

## 🔮 Future Improvements

Bi-directional LSTMs: To capture "stroke context" from both top-to-bottom and bottom-to-top.

CTC Loss Integration: Move from single-character recognition to full word/sentence generation.

Data Augmentation: Injecting "Elastic Distortions" to simulate natural hand tremors and pen pressure.

## 👤 Author

Muhammad Abdulkareem

Aspiring Data Scientist & ML Engineer
