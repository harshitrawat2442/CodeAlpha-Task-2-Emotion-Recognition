import os
import argparse

import numpy as np
import librosa
import torch
import torch.nn as nn


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "emotion_model.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LENGTH = 174


# ============================================================
# CNN MODEL
# Same architecture as train.py
# ============================================================

class EmotionCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.20),

            # Block 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.30),

            # Final pooling
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.40),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}\n\n"
            "Please run train.py first."
        )

    print("\nLoading trained model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    class_names = checkpoint["class_names"]

    num_classes = checkpoint["num_classes"]

    model = EmotionCNN(
        num_classes
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Normalization values saved by train.py
    mean = checkpoint.get(
        "mean",
        0.0
    )

    std = checkpoint.get(
        "std",
        1.0
    )

    print("✓ Model loaded successfully")

    print(
        f"✓ Classes: {num_classes}"
    )

    print(
        f"✓ Device: {DEVICE}"
    )

    return (
        model,
        class_names,
        mean,
        std
    )


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(audio_path):

    if not os.path.exists(audio_path):

        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    print(
        f"\nLoading audio:"
        f"\n{audio_path}"
    )

    audio, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:

        raise ValueError(
            "The audio file is empty."
        )

    print(
        f"✓ Sample rate: {sample_rate}"
    )

    print(
        f"✓ Duration: "
        f"{len(audio) / sample_rate:.2f} seconds"
    )

    return audio


# ============================================================
# EXTRACT MFCC
# ============================================================

def extract_mfcc(audio):

    mfcc = librosa.feature.mfcc(

        y=audio,

        sr=SAMPLE_RATE,

        n_mfcc=N_MFCC
    )

    # --------------------------------------------------------
    # Fix time dimension
    # --------------------------------------------------------

    if mfcc.shape[1] < MAX_LENGTH:

        padding = (
            MAX_LENGTH
            - mfcc.shape[1]
        )

        mfcc = np.pad(
            mfcc,
            (
                (0, 0),
                (0, padding)
            ),
            mode="constant"
        )

    else:

        mfcc = mfcc[
            :,
            :MAX_LENGTH
        ]

    return mfcc.astype(
        np.float32
    )


# ============================================================
# PREPROCESS AUDIO
# ============================================================

def preprocess(
    mfcc,
    mean,
    std
):

    # Same normalization as training

    mfcc = (
        mfcc - mean
    ) / (
        std + 1e-8
    )

    # Add:
    # Batch dimension
    # Channel dimension

    mfcc = mfcc[
        np.newaxis,
        np.newaxis,
        :,
        :
    ]

    tensor = torch.tensor(
        mfcc,
        dtype=torch.float32
    )

    tensor = tensor.to(
        DEVICE
    )

    return tensor


# ============================================================
# PREDICT EMOTION
# ============================================================

def predict(
    model,
    audio_tensor,
    class_names
):

    model.eval()

    with torch.no_grad():

        outputs = model(
            audio_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        predicted_index = torch.argmax(
            probabilities
        ).item()

    predicted_emotion = class_names[
        predicted_index
    ]

    probabilities = (
        probabilities
        .cpu()
        .numpy()
    )

    return (
        predicted_emotion,
        probabilities
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    audio_path,
    emotion,
    probabilities,
    class_names
):

    predicted_index = class_names.index(
        emotion
    )

    confidence = (
        probabilities[predicted_index]
        * 100
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "             SPEECH EMOTION PREDICTION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAudio File:"
        f"\n  {audio_path}"
    )

    print(
        f"\nPredicted Emotion:"
        f"\n  >>> {emotion.upper()} <<<"
    )

    print(
        f"\nConfidence:"
        f"\n  {confidence:.2f}%"
    )

    print(
        "\n"
        + "-" * 70
    )

    print(
        "Emotion Probability Distribution"
    )

    print(
        "-" * 70
    )

    # Sort from highest to lowest

    results = sorted(
        zip(
            class_names,
            probabilities
        ),
        key=lambda item: item[1],
        reverse=True
    )

    for emotion_name, probability in results:

        percentage = (
            probability * 100
        )

        bar_length = int(
            percentage / 3
        )

        bar = "█" * bar_length

        print(
            f"{emotion_name:<12} "
            f"{percentage:6.2f}% "
            f"{bar}"
        )

    print(
        "\n"
        + "=" * 70
    )


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def run_prediction(audio_path):

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        model,
        class_names,
        mean,
        std

    ) = load_model()

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio = load_audio(
        audio_path
    )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    print(
        "\nExtracting MFCC features..."
    )

    mfcc = extract_mfcc(
        audio
    )

    print(
        f"✓ MFCC shape: {mfcc.shape}"
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    audio_tensor = preprocess(
        mfcc,
        mean,
        std
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print(
        "\nRunning model prediction..."
    )

    (
        emotion,
        probabilities

    ) = predict(
        model,
        audio_tensor,
        class_names
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(
        audio_path,
        emotion,
        probabilities,
        class_names
    )

    return emotion, probabilities


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Speech Emotion Recognition "
            "Prediction"
        )
    )

    parser.add_argument(
        "audio",
        nargs="?",
        help="Path to audio file"
    )

    args = parser.parse_args()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "        SPEECH EMOTION RECOGNITION"
    )

    print(
        "                  PREDICT"
    )

    print(
        "=" * 70
    )

    try:

        # ----------------------------------------------------
        # If audio path is not supplied
        # ----------------------------------------------------

        if args.audio:

            audio_path = args.audio

        else:

            audio_path = input(
                "\nEnter audio file path: "
            ).strip()

        # Windows copied paths
        audio_path = audio_path.strip(
            '"'
        ).strip(
            "'"
        )

        if not audio_path:

            print(
                "\n❌ No audio file provided."
            )

            return

        # ----------------------------------------------------
        # Run prediction
        # ----------------------------------------------------

        run_prediction(
            audio_path
        )

    except FileNotFoundError as error:

        print(
            f"\n❌ ERROR:\n{error}"
        )

    except Exception as error:

        print(
            "\n❌ Prediction failed:"
        )

        print(
            error
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()