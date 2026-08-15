import os
import json

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from feature_extraction import create_feature_dataset
from train import EmotionCNN


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/RAVDESS"

MODEL_PATH = "models/emotion_model.pth"

LABEL_PATH = "models/emotion_labels.json"

CACHE_PATH = "outputs/mfcc_cache.npz"

OUTPUT_DIR = "outputs"


CONFUSION_MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

CLASS_ACCURACY_PATH = os.path.join(
    OUTPUT_DIR,
    "class_accuracy.png"
)

PREDICTION_DISTRIBUTION_PATH = os.path.join(
    OUTPUT_DIR,
    "prediction_distribution.png"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.15

BATCH_SIZE = 64


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_features():

    if os.path.exists(
        CACHE_PATH
    ):

        print(
            "\nLoading cached MFCC features..."
        )

        data = np.load(
            CACHE_PATH,
            allow_pickle=True
        )

        return (
            data["X"],
            data["y"]
        )


    print(
        "\nMFCC cache not found."
    )

    print(
        "Extracting features..."
    )


    X, y = create_feature_dataset(
        DATASET_PATH
    )


    np.savez_compressed(
        CACHE_PATH,
        X=X,
        y=y
    )


    return X, y


# ============================================================
# PREPARE TEST DATA
# ============================================================

def prepare_test_data(
    X,
    y,
    checkpoint
):

    class_names = checkpoint[
        "class_names"
    ]


    label_to_index = {

        emotion: index

        for index, emotion
        in enumerate(
            class_names
        )
    }


    y_encoded = np.array([

        label_to_index[label]

        for label in y

    ])


    # CNN channel

    X = X.astype(
        np.float32
    )


    X = X[
        :,
        np.newaxis,
        :,
        :
    ]


    # SAME normalization as training

    mean = checkpoint[
        "mean"
    ]

    std = checkpoint[
        "std"
    ]


    X = (
        X - mean
    ) / (
        std + 1e-8
    )


    # SAME test split

    _, X_test, _, y_test = train_test_split(

        X,

        y_encoded,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y_encoded
    )


    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )


    y_test = torch.tensor(
        y_test,
        dtype=torch.long
    )


    return (
        X_test,
        y_test
    )


# ============================================================
# PREDICTIONS
# ============================================================

def get_predictions(
    model,
    loader
):

    model.eval()


    predictions = []

    actual = []


    total_loss = 0.0

    total = 0


    criterion = nn.CrossEntropyLoss()


    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                inputs
            )


            loss = criterion(
                outputs,
                labels
            )


            total_loss += (
                loss.item()
                *
                inputs.size(0)
            )


            total += inputs.size(0)


            preds = torch.argmax(
                outputs,
                dim=1
            )


            predictions.extend(
                preds.cpu().numpy()
            )


            actual.extend(
                labels.cpu().numpy()
            )


    loss = (
        total_loss / total
    )


    return (
        np.array(actual),
        np.array(predictions),
        loss
    )


# ============================================================
# GRAPH 1
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix_graph(
    y_true,
    y_pred,
    class_names
):

    print(
        "\nCreating confusion matrix..."
    )


    cm = confusion_matrix(
        y_true,
        y_pred
    )


    fig, ax = plt.subplots(
        figsize=(10, 8)
    )


    image = ax.imshow(
        cm,
        interpolation="nearest"
    )


    ax.set_title(
        "Speech Emotion Recognition\nConfusion Matrix",
        fontsize=16
    )


    fig.colorbar(
        image,
        ax=ax
    )


    tick_marks = np.arange(
        len(class_names)
    )


    ax.set_xticks(
        tick_marks
    )

    ax.set_yticks(
        tick_marks
    )


    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        class_names
    )


    ax.set_xlabel(
        "Predicted Emotion"
    )

    ax.set_ylabel(
        "Actual Emotion"
    )


    # Values

    threshold = (
        cm.max() / 2
    )


    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            ax.text(

                j,
                i,

                str(
                    cm[i, j]
                ),

                ha="center",

                va="center",

                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),

                fontsize=11
            )


    plt.tight_layout()


    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=250,
        bbox_inches="tight"
    )


    print(
        f"✓ Saved:"
        f"\n  {CONFUSION_MATRIX_PATH}"
    )


    # SHOW GRAPH

    plt.show()


    plt.close()


# ============================================================
# GRAPH 2
# CLASS ACCURACY
# ============================================================

def create_class_accuracy_graph(
    y_true,
    y_pred,
    class_names
):

    print(
        "\nCreating class accuracy graph..."
    )


    accuracies = []


    for class_index in range(
        len(class_names)
    ):

        mask = (
            y_true == class_index
        )


        total = mask.sum()


        if total == 0:

            accuracy = 0

        else:

            correct = (
                y_pred[mask]
                ==
                class_index
            ).sum()


            accuracy = (
                correct / total
            )


        accuracies.append(
            accuracy
        )


    fig, ax = plt.subplots(
        figsize=(11, 6)
    )


    bars = ax.bar(
        class_names,
        accuracies
    )


    ax.set_title(
        "Emotion-wise Classification Accuracy",
        fontsize=16
    )


    ax.set_xlabel(
        "Emotion"
    )


    ax.set_ylabel(
        "Accuracy"
    )


    ax.set_ylim(
        0,
        1
    )


    ax.grid(
        axis="y",
        alpha=0.3
    )


    plt.xticks(
        rotation=30,
        ha="right"
    )


    # Percentage labels

    for bar, accuracy in zip(
        bars,
        accuracies
    ):

        ax.text(

            bar.get_x()
            +
            bar.get_width()
            / 2,

            accuracy + 0.025,

            f"{accuracy * 100:.1f}%",

            ha="center",

            fontsize=10
        )


    plt.tight_layout()


    plt.savefig(
        CLASS_ACCURACY_PATH,
        dpi=250,
        bbox_inches="tight"
    )


    print(
        f"✓ Saved:"
        f"\n  {CLASS_ACCURACY_PATH}"
    )


    plt.show()


    plt.close()


# ============================================================
# GRAPH 3
# PREDICTION DISTRIBUTION
# ============================================================

def create_prediction_distribution(
    y_true,
    y_pred,
    class_names
):

    print(
        "\nCreating prediction distribution..."
    )


    actual_counts = np.bincount(

        y_true,

        minlength=len(
            class_names
        )
    )


    predicted_counts = np.bincount(

        y_pred,

        minlength=len(
            class_names
        )
    )


    x = np.arange(
        len(class_names)
    )


    width = 0.35


    fig, ax = plt.subplots(
        figsize=(11, 6)
    )


    ax.bar(

        x - width / 2,

        actual_counts,

        width,

        label="Actual"
    )


    ax.bar(

        x + width / 2,

        predicted_counts,

        width,

        label="Predicted"
    )


    ax.set_title(
        "Actual vs Predicted Emotion Distribution",
        fontsize=16
    )


    ax.set_xlabel(
        "Emotion"
    )


    ax.set_ylabel(
        "Number of Samples"
    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        class_names,
        rotation=30,
        ha="right"
    )


    ax.legend()


    ax.grid(
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        PREDICTION_DISTRIBUTION_PATH,
        dpi=250,
        bbox_inches="tight"
    )


    print(
        f"✓ Saved:"
        f"\n  {PREDICTION_DISTRIBUTION_PATH}"
    )


    plt.show()


    plt.close()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def create_report(
    y_true,
    y_pred,
    class_names
):

    report = classification_report(

        y_true,

        y_pred,

        labels=list(
            range(
                len(class_names)
            )
        ),

        target_names=class_names,

        digits=4,

        zero_division=0
    )


    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SPEECH EMOTION RECOGNITION\n"
        )

        file.write(
            "=" * 65
        )

        file.write(
            "\n\n"
        )

        file.write(
            report
        )


    return report


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 70
    )

    print(
        "          SPEECH EMOTION RECOGNITION"
    )

    print(
        "                 MODEL EVALUATION"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "\n❌ Model not found!"
        )

        print(
            f"Expected:"
            f"\n{MODEL_PATH}"
        )

        print(
            "\nRun train.py first."
        )

        raise SystemExit


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\n[1/5] Loading trained model..."
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


    print(
        "✓ Model loaded successfully"
    )


    # ========================================================
    # DATA
    # ========================================================

    print(
        "\n[2/5] Loading test data..."
    )


    X, y = load_features()


    X_test, y_test = prepare_test_data(

        X,

        y,

        checkpoint
    )


    test_dataset = TensorDataset(

        X_test,

        y_test
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False
    )


    print(
        f"✓ Test samples: "
        f"{len(X_test)}"
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    print(
        "\n[3/5] Generating predictions..."
    )


    y_true, y_pred, test_loss = get_predictions(

        model,

        test_loader
    )


    test_accuracy = accuracy_score(
        y_true,
        y_pred
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "                  MODEL RESULTS"
    )

    print(
        "=" * 70
    )


    print(
        f"\nTest Loss     : "
        f"{test_loss:.4f}"
    )


    print(
        f"Test Accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )


    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n[4/5] Classification report..."
    )


    report = create_report(

        y_true,

        y_pred,

        class_names
    )


    print(
        "\n"
        + report
    )


    print(
        f"✓ Report saved:"
        f"\n  {REPORT_PATH}"
    )


    # ========================================================
    # GRAPHS
    # ========================================================

    print(
        "\n[5/5] Generating separate graphs..."
    )


    create_confusion_matrix_graph(

        y_true,

        y_pred,

        class_names
    )


    create_class_accuracy_graph(

        y_true,

        y_pred,

        class_names
    )


    create_prediction_distribution(

        y_true,

        y_pred,

        class_names
    )


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "           EVALUATION COMPLETED"
    )

    print(
        "=" * 70
    )


    print(
        "\nGenerated files:"
    )


    print(
        "\n✓ confusion_matrix.png"
    )

    print(
        "✓ class_accuracy.png"
    )

    print(
        "✓ prediction_distribution.png"
    )

    print(
        "✓ classification_report.txt"
    )


    print(
        "\nAll files are inside:"
    )

    print(
        "outputs/"
    )


    print(
        "\n"
        + "=" * 70
    )