import os
import pandas as pd


# ==============================
# CONFIGURATION
# ==============================

DATASET_PATH = "dataset/RAVDESS"


# RAVDESS emotion codes
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# ==============================
# LOAD DATASET
# ==============================

def load_dataset(dataset_path=DATASET_PATH):

    audio_files = []
    emotions = []
    actors = []

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}"
        )

    for actor_folder in sorted(os.listdir(dataset_path)):

        actor_path = os.path.join(dataset_path, actor_folder)

        if not os.path.isdir(actor_path):
            continue

        for file_name in sorted(os.listdir(actor_path)):

            if not file_name.lower().endswith(".wav"):
                continue

            file_path = os.path.join(actor_path, file_name)

            # RAVDESS filename format:
            # 03-01-05-01-02-01-12.wav
            parts = file_name.split("-")

            if len(parts) < 3:
                continue

            emotion_code = parts[2]

            if emotion_code not in EMOTION_MAP:
                continue

            emotion = EMOTION_MAP[emotion_code]

            audio_files.append(file_path)
            emotions.append(emotion)
            actors.append(actor_folder)

    dataframe = pd.DataFrame({
        "file_path": audio_files,
        "emotion": emotions,
        "actor": actors
    })

    return dataframe


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    df = load_dataset()

    print("\n" + "=" * 50)
    print("RAVDESS DATASET INFORMATION")
    print("=" * 50)

    print(f"\nTotal audio files: {len(df)}")

    print("\nEmotion Distribution:")
    print(df["emotion"].value_counts())

    print("\nActor Distribution:")
    print(df["actor"].value_counts().sort_index())

    print("\nFirst 10 records:")
    print(df.head(10))

    print("\nDataset Shape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\n" + "=" * 50)
    print("Dataset loaded successfully!")
    print("=" * 50)