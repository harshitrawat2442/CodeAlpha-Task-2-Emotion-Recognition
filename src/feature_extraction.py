import os
import numpy as np
import librosa

from data_loader import load_dataset


# ==============================
# CONFIGURATION
# ==============================

SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40

MAX_LENGTH = SAMPLE_RATE * DURATION


# ==============================
# LOAD AND PREPROCESS AUDIO
# ==============================

def load_audio(file_path):
    """
    Load audio file and convert it
    to a fixed length signal.
    """

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    # Normalize audio
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # Make all audio files same length
    if len(audio) < MAX_LENGTH:

        audio = np.pad(
            audio,
            (0, MAX_LENGTH - len(audio)),
            mode="constant"
        )

    else:

        audio = audio[:MAX_LENGTH]

    return audio, sr


# ==============================
# EXTRACT MFCC FEATURES
# ==============================

def extract_mfcc(file_path):
    """
    Extract MFCC features from an audio file.
    """

    audio, sr = load_audio(file_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Normalize MFCC
    mfcc = (mfcc - np.mean(mfcc)) / (
        np.std(mfcc) + 1e-8
    )

    return mfcc


# ==============================
# CREATE FEATURE DATASET
# ==============================

def create_feature_dataset(dataset_path="dataset/RAVDESS"):

    df = load_dataset(dataset_path)

    features = []
    labels = []

    print("\nExtracting MFCC features...\n")

    for index, row in df.iterrows():

        try:

            mfcc = extract_mfcc(row["file_path"])

            features.append(mfcc)
            labels.append(row["emotion"])

        except Exception as error:

            print(
                f"Error processing: "
                f"{row['file_path']}"
            )

            print(error)

    X = np.array(features)
    y = np.array(labels)

    return X, y


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    X, y = create_feature_dataset()

    print("\n" + "=" * 50)
    print("MFCC FEATURE EXTRACTION")
    print("=" * 50)

    print(f"\nFeature shape: {X.shape}")
    print(f"Label shape: {y.shape}")

    print(f"\nNumber of samples: {len(X)}")

    print(f"\nNumber of MFCC coefficients: {X.shape[1]}")

    print(f"\nMFCC time steps: {X.shape[2]}")

    print("\nEmotion labels:")

    unique_labels, counts = np.unique(
        y,
        return_counts=True
    )

    for label, count in zip(unique_labels, counts):

        print(f"{label}: {count}")

    print("\n" + "=" * 50)
    print("MFCC extraction completed successfully!")
    print("=" * 50)