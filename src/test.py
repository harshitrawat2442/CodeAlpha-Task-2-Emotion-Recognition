import os
import json
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

LABEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "emotion_labels.json"
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
# ============================================================

class EmotionCNN(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()


        self.features = nn.Sequential(

            # Block 1

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            nn.Dropout2d(
                0.20
            ),


            # Block 2

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            nn.Dropout2d(
                0.25
            ),


            # Block 3

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            nn.Dropout2d(
                0.30
            ),


            nn.AdaptiveAvgPool2d(
                (4, 4)
            )
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.40
            ),

            nn.Linear(
                256,
                num_classes
            )
        )


    def forward(
        self,
        x
    ):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}\n\n"
            "Run train.py first."
        )


    checkpoint = torch.load(

        MODEL_PATH,

        map_location=DEVICE,

        weights_only=False
    )


    class_names = checkpoint[
        "class_names"
    ]


    num_classes = checkpoint[
        "num_classes"
    ]


    model = EmotionCNN(
        num_classes
    ).to(
        DEVICE
    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    model.eval()


    mean = checkpoint.get(
        "mean",
        0.0
    )


    std = checkpoint.get(
        "std",
        1.0
    )


    return (
        model,
        class_names,
        mean,
        std
    )


# ============================================================
# EXTRACT MFCC
# ============================================================

def extract_mfcc(
    audio_path
):

    print(
        "\nLoading audio..."
    )


    audio, sr = librosa.load(

        audio_path,

        sr=SAMPLE_RATE,

        mono=True
    )


    if len(audio) == 0:

        raise ValueError(
            "Audio file is empty."
        )


    # Normalize audio

    max_value = np.max(
        np.abs(audio)
    )


    if max_value > 0:

        audio = (
            audio / max_value
        )


    # MFCC

    mfcc = librosa.feature.mfcc(

        y=audio,

        sr=sr,

        n_mfcc=N_MFCC
    )


    # Fix time dimension

    if mfcc.shape[1] < MAX_LENGTH:

        pad_width = (
            MAX_LENGTH
            -
            mfcc.shape[1]
        )


        mfcc = np.pad(

            mfcc,

            (
                (0, 0),
                (0, pad_width)
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
# PREDICT
# ============================================================

def predict_emotion(
    model,
    class_names,
    mfcc,
    mean,
    std
):

    # Add channel dimension

    mfcc = mfcc[
        np.newaxis,
        np.newaxis,
        :,
        :
    ]


    # Normalize exactly like training

    mfcc = (

        mfcc - mean

    ) / (

        std + 1e-8

    )


    tensor = torch.tensor(

        mfcc,

        dtype=torch.float32

    ).to(
        DEVICE
    )


    with torch.no_grad():

        outputs = model(
            tensor
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

    print(
        "\n"
        + "=" * 65
    )

    print(
        "              SPEECH EMOTION RESULT"
    )

    print(
        "=" * 65
    )


    print(
        f"\nAudio File:"
    )

    print(
        f"  {audio_path}"
    )


    print(
        f"\nPredicted Emotion:"
    )

    print(
        f"  >>> {emotion.upper()} <<<"
    )


    print(
        "\nConfidence:"
    )


    confidence = (

        probabilities[
            class_names.index(
                emotion
            )
        ]
        * 100

    )


    print(
        f"  {confidence:.2f}%"
    )


    print(
        "\nAll Emotion Probabilities:"
    )


    # Sort highest → lowest

    results = sorted(

        zip(
            class_names,
            probabilities
        ),

        key=lambda x: x[1],

        reverse=True
    )


    for emotion_name, probability in results:

        bar_length = int(
            probability * 30
        )


        bar = (
            "█"
            *
            bar_length
        )


        print(

            f"  {emotion_name:<12} "
            f"{probability * 100:6.2f}% "
            f"{bar}"

        )


    print(
        "\n"
        + "=" * 65
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Speech Emotion Recognition "
            "using trained CNN model"
        )
    )


    parser.add_argument(

        "audio",

        nargs="?",

        help=(
            "Path to audio file "
            "(WAV recommended)"
        )
    )


    args = parser.parse_args()


    # ========================================================
    # HEADER
    # ========================================================

    print(
        "\n"
        + "=" * 65
    )

    print(
        "        SPEECH EMOTION RECOGNITION"
    )

    print(
        "                 TEST"
    )

    print(
        "=" * 65
    )


    print(
        f"\nDevice: {DEVICE}"
    )


    # ========================================================
    # AUDIO PATH
    # ========================================================

    audio_path = args.audio


    if audio_path is None:

        audio_path = input(

            "\nEnter audio file path: "

        ).strip()


    # Remove quotes if copied from Windows

    audio_path = audio_path.strip(
        '"'
    ).strip(
        "'"
    )


    if not os.path.exists(
        audio_path
    ):

        print(
            f"\n❌ Audio file not found:"
            f"\n{audio_path}"
        )

        return


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading trained model..."
    )


    try:

        (
            model,
            class_names,
            mean,
            std

        ) = load_model()


    except Exception as error:

        print(
            "\n❌ Model loading failed:"
        )

        print(
            error
        )

        return


    print(
        "✓ Model loaded successfully"
    )


    print(
        f"✓ Classes: "
        f"{len(class_names)}"
    )


    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    try:

        mfcc = extract_mfcc(
            audio_path
        )


        print(
            f"✓ MFCC shape: "
            f"{mfcc.shape}"
        )


    except Exception as error:

        print(
            "\n❌ Feature extraction failed:"
        )

        print(
            error
        )

        return


    # ========================================================
    # PREDICTION
    # ========================================================

    print(
        "\nRunning prediction..."
    )


    try:

        (
            emotion,
            probabilities

        ) = predict_emotion(

            model,

            class_names,

            mfcc,

            mean,

            std

        )


    except Exception as error:

        print(
            "\n❌ Prediction failed:"
        )

        print(
            error
        )

        return


    # ========================================================
    # RESULTS
    # ========================================================

    display_results(

        audio_path,

        emotion,

        probabilities,

        class_names

    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()