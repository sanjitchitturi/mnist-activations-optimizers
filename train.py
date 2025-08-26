import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# =========================
# Setup
# =========================
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# =========================
# Load Dataset
# =========================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# =========================
# Experiment Configs
# =========================
activations = ["relu", "sigmoid", "tanh"]
optimizers = ["sgd", "adam", "rmsprop"]

results = []

# =========================
# Model Builder
# =========================
def build_model(activation, optimizer):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=activation),
        Dense(64, activation=activation),
        Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =========================
# Run Experiments with Progress Bar
# =========================
best_acc = 0.0
best_model = None
best_name = None

exp_combos = [(opt, act) for opt in optimizers for act in activations]

for opt, act in tqdm(exp_combos, desc="Running Experiments"):
    print(f"\nTraining with Optimizer={opt}, Activation={act} ...")

    model = build_model(act, opt)
    history = model.fit(
        x_train, y_train_cat,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        verbose=1
    )

    train_acc = history.history['accuracy'][-1]
    train_loss = history.history['loss'][-1]

    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)

    results.append([opt, act, train_acc, test_acc, train_loss, test_loss])

    # Save model
    model_path = os.path.join(MODELS_DIR, f"model_{opt}_{act}.h5")
    model.save(model_path)

    # Track best model
    if test_acc > best_acc:
        best_acc = test_acc
        best_model = model
        best_name = f"{opt}_{act}"

# =========================
# Save Results Table
# =========================
df = pd.DataFrame(results, columns=["Optimizer", "Activation", "Train Accuracy", "Test Accuracy", "Train Loss", "Test Loss"])
df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)

# =========================
# Plot Bar Chart of Test Accuracy
# =========================
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Optimizer", y="Test Accuracy", hue="Activation")
plt.title("Test Accuracy by Optimizer & Activation")
plt.savefig(os.path.join(RESULTS_DIR, "results.png"))
plt.close()

# =========================
# Heatmap
# =========================
heatmap_data = df.pivot("Optimizer", "Activation", "Test Accuracy")
plt.figure(figsize=(6, 4))
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="Blues")
plt.title("Accuracy Heatmap")
plt.savefig(os.path.join(RESULTS_DIR, "results_heatmap.png"))
plt.close()

# =========================
# Confusion Matrix for Best Model
# =========================
y_pred = best_model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 8))
ConfusionMatrixDisplay(cm).plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix (Best Model: {best_name})")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

# =========================
# Print Summary Table
# =========================
print("\nFinal Results Summary:")
print(df.sort_values(by="Test Accuracy", ascending=False).to_string(index=False))

print("\nExperiment complete!")
print(f"Best model: {best_name} with Test Accuracy = {best_acc:.4f}")
print("Results saved in ./results and models in ./models")
