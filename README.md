# 🎙️ Speech Emotion Recognition

A Machine Learning project that detects human emotions from
speech audio using MFCC feature extraction and a Convolutional
Neural Network (CNN).

The project uses the RAVDESS speech emotion dataset and provides
a complete pipeline from audio data exploration and feature
extraction to model training, evaluation and final prediction.

---

## 📌 Project Overview

Speech Emotion Recognition (SER) is a Machine Learning task that
identifies the emotional state expressed in a person's speech.

In this project, audio signals are processed using MFCC
(Mel-Frequency Cepstral Coefficients) and supplied to a CNN-based
classification model.

The complete workflow includes:

Audio Dataset
→ Data Exploration
→ MFCC Feature Extraction
→ CNN Model Training
→ Model Evaluation
→ Emotion Prediction

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze and explore speech emotion data
- Process raw audio files
- Extract meaningful MFCC features
- Build a CNN-based emotion classification model
- Train the model on speech emotion data
- Evaluate model performance
- Generate confusion matrix and classification reports
- Visualize training and evaluation results
- Predict emotions from new audio files

---

## 🛠️ Tech Used

### Programming Language

- Python 3.14

### Machine Learning / Deep Learning

- PyTorch
- Convolutional Neural Network (CNN)

### Audio Processing

- Librosa
- SoundFile

### Data Processing

- NumPy
- Pandas

### Visualization

- Matplotlib

### Development Environment

- Jupyter Notebook
- VS Code

### Dataset

- RAVDESS
  (Ryerson Audio-Visual Database of Emotional Speech and Song)

---

## 📄 Website Pages

This is a Machine Learning / Speech Emotion Recognition
project, so it does not contain traditional website pages.

Instead, the project is organized into notebooks and Python
modules for the complete ML workflow.

### Main Notebooks

1. `01_Data_Exploration.ipynb`
2. `02_Feature_Extraction.ipynb`
3. `03_Model_Training.ipynb`
4. `04_Model_Evaluation.ipynb`
5. `05_Prediction_Demo.ipynb`

---

## ✨ Features

- 🎙️ Speech audio processing
- 🔊 Audio playback
- 📊 Dataset exploration
- 📈 Emotion distribution visualization
- 🎵 Audio waveform visualization
- 🧠 MFCC feature extraction
- 🔥 CNN-based emotion classification
- 📉 Training loss visualization
- 📈 Training accuracy visualization
- 📊 Confusion matrix
- 📋 Classification report
- 🎯 Per-emotion performance analysis
- 🔮 New audio emotion prediction
- 📊 Prediction probability visualization
- 💾 Saved trained model
- 💾 Saved prediction results

---

## 📁 Project Structure

```text
Emotion_Recognition/
│
├── dataset/
│   └── RAVDESS/
│       ├── Actor_01/
│       ├── Actor_02/
│       ├── ...
│       └── Actor_24/
│
├── models/
│   ├── emotion_model.pth
│   ├── emotion_model.keras
│   └── emotion_labels.json
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Feature_Extraction.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_Model_Evaluation.ipynb
│   └── 05_Prediction_Demo.ipynb
│
├── outputs/
│   ├── class_accuracy.png
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── emotion_distribution.png
│   ├── evaluation_summary.csv
│   ├── mfcc_features.npy
│   ├── training_accuracy.png
│   ├── training_loss.png
│   ├── training_history.csv
│   └── ...
│
├── src/
│   ├── data_loader.py
│   ├── feature_extraction.py
│   ├── visualization.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── test.py
│
└── README.md
```
## 🔧 What I Did
**Developed a complete Speech Emotion Recognition project using the RAVDESS dataset.
Performed data exploration and visualization to understand the speech emotion classes.
Extracted MFCC (Mel-Frequency Cepstral Coefficients) features from the audio files.
Built and trained a CNN (Convolutional Neural Network) for emotion classification.
Evaluated the trained model using accuracy, classification report, confusion matrix, and class-wise performance.
Created visualization graphs for training accuracy, training loss, MFCCs, emotion distribution, and model evaluation.
Developed a prediction system that takes a new .wav audio file and predicts the emotion with a confidence score.
**---
## 🤖 Machine Learning Models
1. Convolutional Neural Network (CNN)
Used as the main Deep Learning model for Speech Emotion Recognition.
Takes extracted MFCC features as input.
Uses convolution, batch normalization, ReLU activation, pooling, and dropout layers.
Learns important patterns from speech features for emotion classification.
The final fully connected layers classify the audio into the supported emotion classes.
The trained model is saved as emotion_model.pth and emotion_model.keras.
---
## 📊 Model Evaluation
**
Evaluated the trained CNN model on the test dataset to measure its classification performance.
Calculated accuracy to determine the overall percentage of correctly classified emotions.
Generated a classification report containing precision, recall, F1-score, and support for each emotion class.
Created a confusion matrix to analyze correct predictions and misclassifications between emotion categories.
Generated class-wise accuracy to compare the model's performance across different emotions.
Visualized training accuracy and training loss to analyze the model's learning performance.
Saved the evaluation results and graphs in the outputs/ folder for further analysis.
**
---
## 📈 Result
Successfully developed a Speech Emotion Recognition system using the RAVDESS dataset.
The trained CNN model successfully learns patterns from MFCC speech features for emotion classification.
Generated training accuracy and loss graphs to visualize the model's learning process.
Generated a confusion matrix and classification report to analyze model performance.
Created class-wise accuracy results to compare performance across different emotion categories.
Successfully implemented a prediction system that can classify the emotion of a new .wav audio file.
All important trained models, graphs, reports, and feature outputs are saved in the models/ and outputs/ folders.
