import os
import csv
import json
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


from feature_extraction import create_feature_dataset


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "RAVDESS"
)


MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "emotion_model.pth"
)


LABEL_PATH = os.path.join(
    MODEL_DIR,
    "emotion_labels.json"
)


FEATURE_CACHE = os.path.join(
    OUTPUT_DIR,
    "mfcc_cache.npz"
)


ACCURACY_PATH = os.path.join(
    OUTPUT_DIR,
    "training_accuracy.png"
)


LOSS_PATH = os.path.join(
    OUTPUT_DIR,
    "training_loss.png"
)


HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "training_history.csv"
)


LOG_PATH = os.path.join(
    OUTPUT_DIR,
    "training_log.txt"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

RANDOM_STATE = 42

EPOCHS = 30

BATCH_SIZE = 64

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

TEST_SIZE = 0.15

VALIDATION_SIZE = 0.15

EARLY_STOPPING_PATIENCE = 6

LR_PATIENCE = 2

LR_FACTOR = 0.5


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


# ============================================================
# CPU OPTIMIZATION
# ============================================================

if DEVICE.type == "cpu":

    cpu_count = os.cpu_count()

    if cpu_count is not None:

        torch.set_num_threads(
            max(
                1,
                cpu_count // 2
            )
        )


# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(
    RANDOM_STATE
)

torch.manual_seed(
    RANDOM_STATE
)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_STATE
    )


# ============================================================
# MATPLOTLIB LIVE MODE
# ============================================================

plt.ion()


# ============================================================
# LIVE GRAPH FUNCTION
# ============================================================

def update_training_graphs(
    train_losses,
    val_losses,
    train_acc,
    val_acc
):

    epoch_numbers = range(
        1,
        len(train_losses) + 1
    )


    # ========================================================
    # ACCURACY WINDOW
    # ========================================================

    plt.figure(
        1,
        figsize=(10, 6)
    )

    plt.clf()


    plt.plot(
        epoch_numbers,
        train_acc,
        marker="o",
        linewidth=2,
        label="Training Accuracy"
    )


    plt.plot(
        epoch_numbers,
        val_acc,
        marker="o",
        linewidth=2,
        label="Validation Accuracy"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )


    plt.title(
        "Speech Emotion Recognition - Accuracy"
    )


    plt.ylim(
        0,
        1
    )


    plt.xticks(
        list(epoch_numbers)
    )


    plt.grid(
        True,
        alpha=0.3
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        ACCURACY_PATH,
        dpi=200,
        bbox_inches="tight"
    )


    plt.draw()

    plt.pause(
        0.05
    )


    # ========================================================
    # LOSS WINDOW
    # ========================================================

    plt.figure(
        2,
        figsize=(10, 6)
    )

    plt.clf()


    plt.plot(
        epoch_numbers,
        train_losses,
        marker="o",
        linewidth=2,
        label="Training Loss"
    )


    plt.plot(
        epoch_numbers,
        val_losses,
        marker="o",
        linewidth=2,
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )


    plt.title(
        "Speech Emotion Recognition - Loss"
    )


    plt.xticks(
        list(epoch_numbers)
    )


    plt.grid(
        True,
        alpha=0.3
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        LOSS_PATH,
        dpi=200,
        bbox_inches="tight"
    )


    plt.draw()

    plt.pause(
        0.05
    )


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

            # =================================================
            # BLOCK 1
            # =================================================

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


            # =================================================
            # BLOCK 2
            # =================================================

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


            # =================================================
            # BLOCK 3
            # =================================================

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


            # =================================================
            # ADAPTIVE POOLING
            # =================================================

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

        x = self.features(
            x
        )

        x = self.classifier(
            x
        )

        return x


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():

    if os.path.exists(
        FEATURE_CACHE
    ):

        print(
            "\n✓ Loading cached MFCC features..."
        )


        data = np.load(
            FEATURE_CACHE,
            allow_pickle=True
        )


        X = data["X"]

        y = data["y"]


        print(
            "✓ MFCC cache loaded"
        )


        return X, y


    print(
        "\n⚡ MFCC cache not found."
    )

    print(
        "Extracting features..."
    )


    X, y = create_feature_dataset(
        DATASET_PATH
    )


    print(
        "\nSaving MFCC cache..."
    )


    np.savez_compressed(
        FEATURE_CACHE,
        X=X,
        y=y
    )


    print(
        "✓ MFCC cache saved"
    )


    return X, y


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader,
    criterion
):

    model.eval()


    total_loss = 0.0

    correct = 0

    total = 0


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


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    average_loss = (
        total_loss
        /
        total
    )


    accuracy = (
        correct
        /
        total
    )


    return (
        average_loss,
        accuracy
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(
    train_losses,
    val_losses,
    train_acc,
    val_acc
):

    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow([

            "epoch",

            "train_loss",

            "validation_loss",

            "train_accuracy",

            "validation_accuracy"

        ])


        for index in range(
            len(train_losses)
        ):

            writer.writerow([

                index + 1,

                train_losses[index],

                val_losses[index],

                train_acc[index],

                val_acc[index]

            ])


# ============================================================
# SAVE LOG
# ============================================================

def write_log(
    message
):

    with open(
        LOG_PATH,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            message + "\n"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    start_time = time.time()


    # ========================================================
    # START LOG
    # ========================================================

    with open(
        LOG_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SPEECH EMOTION RECOGNITION\n"
        )

        file.write(
            "=" * 70
            + "\n"
        )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "        SPEECH EMOTION RECOGNITION"
    )

    print(
        "              FULL CNN TRAINING"
    )

    print(
        "=" * 70
    )


    print(
        f"\nDevice       : {DEVICE}"
    )

    print(
        f"Epochs       : {EPOCHS}"
    )

    print(
        f"Batch Size   : {BATCH_SIZE}"
    )

    print(
        f"Learning Rate: {LEARNING_RATE}"
    )

    print(
        f"Dataset      : {DATASET_PATH}"
    )


    write_log(
        f"Device: {DEVICE}"
    )


    # ========================================================
    # STEP 1
    # ========================================================

    print(
        "\n[1/7] Loading features..."
    )


    X, y = load_features()


    print(
        f"Feature shape: {X.shape}"
    )

    print(
        f"Samples      : {len(X)}"
    )


    # ========================================================
    # STEP 2
    # ========================================================

    print(
        "\n[2/7] Encoding labels..."
    )


    encoder = LabelEncoder()


    y_encoded = encoder.fit_transform(
        y
    )


    class_names = list(
        encoder.classes_
    )


    num_classes = len(
        class_names
    )


    print(
        f"\nClasses: {num_classes}"
    )


    for index, emotion in enumerate(
        class_names
    ):

        print(
            f"  {index} -> {emotion}"
        )


    # ========================================================
    # SAVE LABELS
    # ========================================================

    label_mapping = {

        str(index): emotion

        for index, emotion
        in enumerate(
            class_names
        )
    }


    with open(
        LABEL_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            label_mapping,
            file,
            indent=4
        )


    print(
        f"\n✓ Labels saved:"
        f"\n  {LABEL_PATH}"
    )


    # ========================================================
    # STEP 3
    # ========================================================

    print(
        "\n[3/7] Preparing CNN data..."
    )


    X = X.astype(
        np.float32
    )


    # Add channel

    X = X[
        :,
        np.newaxis,
        :,
        :
    ]


    # ========================================================
    # NORMALIZATION
    # ========================================================

    mean = float(
        X.mean()
    )


    std = float(
        X.std()
    )


    X = (
        X - mean
    ) / (
        std + 1e-8
    )


    print(
        f"Mean : {mean:.6f}"
    )

    print(
        f"Std  : {std:.6f}"
    )


    # ========================================================
    # STEP 4
    # ========================================================

    print(
        "\n[4/7] Splitting dataset..."
    )


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y_encoded,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y_encoded
    )


    validation_ratio = (

        VALIDATION_SIZE
        /
        (1 - TEST_SIZE)

    )


    X_train, X_val, y_train, y_val = train_test_split(

        X_train,

        y_train,

        test_size=validation_ratio,

        random_state=RANDOM_STATE,

        stratify=y_train
    )


    print(
        f"Training   : {len(X_train)}"
    )

    print(
        f"Validation : {len(X_val)}"
    )

    print(
        f"Testing    : {len(X_test)}"
    )


    # ========================================================
    # TENSORS
    # ========================================================

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )


    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )


    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )


    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )


    y_val = torch.tensor(
        y_val,
        dtype=torch.long
    )


    y_test = torch.tensor(
        y_test,
        dtype=torch.long
    )


    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = TensorDataset(
        X_train,
        y_train
    )


    val_dataset = TensorDataset(
        X_val,
        y_val
    )


    test_dataset = TensorDataset(
        X_test,
        y_test
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=(
            DEVICE.type == "cuda"
        )
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=(
            DEVICE.type == "cuda"
        )
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=(
            DEVICE.type == "cuda"
        )
    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    print(
        "\nCalculating class weights..."
    )


    classes = np.unique(
        y_train.numpy()
    )


    weights = compute_class_weight(

        class_weight="balanced",

        classes=classes,

        y=y_train.numpy()
    )


    class_weights = torch.tensor(

        weights,

        dtype=torch.float32

    ).to(
        DEVICE
    )


    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\n[5/7] Building CNN..."
    )


    model = EmotionCNN(
        num_classes
    ).to(
        DEVICE
    )


    # ========================================================
    # MODEL PARAMETERS
    # ========================================================

    total_parameters = sum(

        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad
    )


    print(
        f"Trainable parameters:"
        f" {total_parameters:,}"
    )


    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss(

        weight=class_weights
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=LR_FACTOR,

        patience=LR_PATIENCE
    )


    print(
        "✓ Model ready"
    )


    # ========================================================
    # HISTORY
    # ========================================================

    train_losses = []

    val_losses = []

    train_accuracies = []

    val_accuracies = []


    best_val_accuracy = 0.0

    best_val_loss = float(
        "inf"
    )

    patience_counter = 0


    # ========================================================
    # TRAINING
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "                 TRAINING STARTED"
    )

    print(
        "=" * 70
    )


    for epoch in range(
        EPOCHS
    ):

        epoch_start = time.time()


        model.train()


        running_loss = 0.0

        correct = 0

        total = 0


        # ====================================================
        # TRAIN BATCHES
        # ====================================================

        for batch_index, (
            inputs,
            labels
        ) in enumerate(
            train_loader
        ):

            inputs = inputs.to(
                DEVICE,
                non_blocking=True
            )


            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            outputs = model(
                inputs
            )


            loss = criterion(
                outputs,
                labels
            )


            loss.backward()


            # Gradient clipping

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=1.0
            )


            optimizer.step()


            running_loss += (

                loss.item()
                *
                inputs.size(0)

            )


            predictions = torch.argmax(

                outputs,

                dim=1

            )


            correct += (

                predictions == labels

            ).sum().item()


            total += labels.size(0)


        # ====================================================
        # TRAIN METRICS
        # ====================================================

        train_loss = (

            running_loss
            /
            total

        )


        train_accuracy = (

            correct
            /
            total

        )


        # ====================================================
        # VALIDATION
        # ====================================================

        val_loss, val_accuracy = evaluate_model(

            model,

            val_loader,

            criterion

        )


        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step(
            val_loss
        )


        # ====================================================
        # HISTORY
        # ====================================================

        train_losses.append(
            train_loss
        )


        val_losses.append(
            val_loss
        )


        train_accuracies.append(
            train_accuracy
        )


        val_accuracies.append(
            val_accuracy
        )


        # ====================================================
        # SAVE CSV
        # ====================================================

        save_history(

            train_losses,

            val_losses,

            train_accuracies,

            val_accuracies

        )


        # ====================================================
        # UPDATE LIVE GRAPHS
        # ====================================================

        update_training_graphs(

            train_losses,

            val_losses,

            train_accuracies,

            val_accuracies

        )


        # ====================================================
        # TIME
        # ====================================================

        epoch_time = (

            time.time()
            -
            epoch_start

        )


        current_lr = (

            optimizer
            .param_groups[0]["lr"]

        )


        # ====================================================
        # PRINT RESULT
        # ====================================================

        print(
            f"\nEpoch "
            f"{epoch + 1:02d}/{EPOCHS}"
        )


        print(
            f"  Train Loss : "
            f"{train_loss:.4f}"
        )


        print(
            f"  Train Acc  : "
            f"{train_accuracy * 100:.2f}%"
        )


        print(
            f"  Val Loss   : "
            f"{val_loss:.4f}"
        )


        print(
            f"  Val Acc    : "
            f"{val_accuracy * 100:.2f}%"
        )


        print(
            f"  LR         : "
            f"{current_lr:.7f}"
        )


        print(
            f"  Time       : "
            f"{epoch_time:.2f}s"
        )


        print(
            "  ✓ Graphs updated"
        )


        # ====================================================
        # WRITE LOG
        # ====================================================

        write_log(

            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Train Acc: {train_accuracy:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Acc: {val_accuracy:.6f} | "
            f"LR: {current_lr:.8f}"
        )


        # ====================================================
        # BEST MODEL
        # ====================================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (

                val_accuracy

            )


            best_val_loss = (

                val_loss

            )


            patience_counter = 0


            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "num_classes":
                    num_classes,

                "class_names":
                    class_names,

                "input_shape":
                    list(
                        X_train.shape[1:]
                    ),

                "mean":
                    mean,

                "std":
                    std,

                "best_val_accuracy":
                    best_val_accuracy,

                "best_val_loss":
                    best_val_loss,

                "epoch":
                    epoch + 1,

                "optimizer_state_dict":
                    optimizer.state_dict()

            }


            torch.save(

                checkpoint,

                MODEL_PATH

            )


            print(
                "  🏆 BEST MODEL SAVED"
            )


        else:

            patience_counter += 1


            print(
                f"  No improvement:"
                f" {patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if patience_counter >= (

            EARLY_STOPPING_PATIENCE

        ):

            print(
                "\n🛑 Early stopping triggered."
            )

            break


    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    total_time = (

        time.time()
        -
        start_time

    )


    minutes = int(

        total_time
        //
        60

    )


    seconds = int(

        total_time
        %
        60

    )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print(
        "\nLoading best model..."
    )


    checkpoint = torch.load(

        MODEL_PATH,

        map_location=DEVICE,

        weights_only=False

    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "                    FINAL TEST"
    )

    print(
        "=" * 70
    )


    test_loss, test_accuracy = evaluate_model(

        model,

        test_loader,

        criterion

    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        f"\nBest Validation Accuracy : "
        f"{best_val_accuracy * 100:.2f}%"
    )


    print(
        f"Final Test Accuracy     : "
        f"{test_accuracy * 100:.2f}%"
    )


    print(
        f"Final Test Loss         : "
        f"{test_loss:.4f}"
    )


    print(
        f"Total Training Time     : "
        f"{minutes}m {seconds}s"
    )


    # ========================================================
    # FINAL GRAPHS
    # ========================================================

    update_training_graphs(

        train_losses,

        val_losses,

        train_accuracies,

        val_accuracies

    )


    # ========================================================
    # SAVE FINAL LOG
    # ========================================================

    write_log(
        ""
    )


    write_log(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )


    write_log(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )


    write_log(
        f"Test Loss: "
        f"{test_loss:.6f}"
    )


    write_log(
        f"Training Time: "
        f"{minutes}m {seconds}s"
    )


    # ========================================================
    # KEEP GRAPH WINDOWS OPEN
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "              TRAINING COMPLETED"
    )

    print(
        "=" * 70
    )


    print(
        "\nGenerated files:"
    )


    print(
        f"\n✓ {MODEL_PATH}"
    )


    print(
        f"✓ {LABEL_PATH}"
    )


    print(
        f"✓ {FEATURE_CACHE}"
    )


    print(
        f"✓ {ACCURACY_PATH}"
    )


    print(
        f"✓ {LOSS_PATH}"
    )


    print(
        f"✓ {HISTORY_PATH}"
    )


    print(
        f"✓ {LOG_PATH}"
    )


    print(
        "\nGraph windows will remain open."
    )


    # Turn interactive mode off only after training

    plt.ioff()

    plt.show()