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
---
## 🔧 What I Did

- Developed a complete **Speech Emotion Recognition** system using the RAVDESS dataset.
- Performed **audio data exploration and visualization** to understand different emotion classes.
- Extracted **MFCC (Mel-Frequency Cepstral Coefficients)** features from speech audio using Librosa.
- Prepared and processed the extracted features for **CNN-based classification**.
- Trained the model using separate **training, validation, and testing datasets**.
- Evaluated the model using **accuracy, loss, confusion matrix, and classification report**.
- Developed a **prediction pipeline** for recognizing emotions from new `.wav` audio files.

---

## 🤖 Machine Learning Models

### Convolutional Neural Network (CNN)

- Used **CNN as the primary Deep Learning model** for speech emotion classification.
- MFCC features are provided as input to the CNN model.
- The architecture includes **Convolution, Batch Normalization, ReLU, Max Pooling, Dropout, and Fully Connected layers**.
- The model learns important speech patterns associated with different emotions.
- The final output layer classifies audio into **8 emotion classes**:
  - Angry
  - Calm
  - Disgust
  - Fearful
  - Happy
  - Neutral
  - Sad
  - Surprised
- The trained model is saved in the `models/` directory.

---

## 📊 Model Evaluation

- Evaluated the trained CNN model using a separate **test dataset**.
- Monitored **training and validation accuracy** during model training.
- Analyzed **training and validation loss** to monitor the learning process.
- Generated a **Confusion Matrix** to analyze correct and incorrect predictions.
- Generated a **Classification Report** containing Precision, Recall, F1-Score, and Support.
- Calculated **class-wise performance** for individual emotion categories.
- Saved evaluation graphs and results in the **`outputs/`** directory.

---

## 📈 Result

- Successfully developed an end-to-end **Speech Emotion Recognition system**.
- The CNN model successfully learned patterns from **MFCC speech features**.
- The system can classify speech into **8 different emotion categories**.
- Generated **training accuracy and loss graphs** to analyze model learning.
- Generated **Confusion Matrix and Classification Report** for detailed evaluation.
- Analyzed the model's performance across different emotion classes.
- Implemented a working **prediction pipeline** for new `.wav` audio files.
- Successfully saved the trained model and evaluation outputs for future predictions.
