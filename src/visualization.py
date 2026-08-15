import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from data_loader import load_dataset
from feature_extraction import extract_mfcc


# ==============================
# CONFIGURATION
# ==============================

DATASET_PATH = "dataset/RAVDESS"
OUTPUT_PATH = "outputs"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ==============================
# EMOTION DISTRIBUTION
# ==============================

def plot_emotion_distribution(df):

    emotion_counts = df["emotion"].value_counts().sort_index()

    plt.figure(figsize=(10, 6))

    emotion_counts.plot(kind="bar")

    plt.title("Emotion Distribution in RAVDESS Dataset")
    plt.xlabel("Emotion")
    plt.ylabel("Number of Audio Files")

    plt.xticks(rotation=45)

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_PATH,
        "emotion_distribution.png"
    )

    plt.savefig(output_file, dpi=300)

    print(f"Saved: {output_file}")

    # Display image
    plt.show()

    plt.close()


# ==============================
# SAMPLE WAVEFORM
# ==============================

def plot_waveform(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=16000
    )

    plt.figure(figsize=(12, 4))

    librosa.display.waveshow(
        audio,
        sr=sr
    )

    plt.title("Sample Speech Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_PATH,
        "sample_waveform.png"
    )

    plt.savefig(output_file, dpi=300)

    print(f"Saved: {output_file}")

    # Display image
    plt.show()

    plt.close()


# ==============================
# MFCC VISUALIZATION
# ==============================

def plot_mfcc(file_path):

    mfcc = extract_mfcc(file_path)

    plt.figure(figsize=(12, 6))

    librosa.display.specshow(
        mfcc,
        x_axis="time",
        cmap="viridis"
    )

    plt.colorbar(
        format="%+2.0f"
    )

    plt.title("MFCC Representation of Speech")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_PATH,
        "sample_mfcc.png"
    )

    plt.savefig(output_file, dpi=300)

    print(f"Saved: {output_file}")

    # Display image
    plt.show()

    plt.close()


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("SPEECH EMOTION VISUALIZATION")
    print("=" * 50)

    # Load dataset
    df = load_dataset(DATASET_PATH)

    print(f"\nTotal audio files: {len(df)}")

    # Emotion distribution
    print("\nCreating emotion distribution...")
    plot_emotion_distribution(df)

    # Select first audio sample
    sample_file = df.iloc[0]["file_path"]

    print(f"\nSample audio:")
    print(sample_file)

    # Waveform
    print("\nCreating waveform...")
    plot_waveform(sample_file)

    # MFCC
    print("\nCreating MFCC visualization...")
    plot_mfcc(sample_file)

    print("\n" + "=" * 50)
    print("VISUALIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 50)