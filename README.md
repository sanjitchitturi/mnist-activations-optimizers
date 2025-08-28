# MNIST Activations & Optimizers

**A compact experiment** that trains 9 MLP models on the MNIST handwritten digit dataset (3 activation functions × 3 optimizers), records metrics, and saves visualizations.

---

## Overview

This repository contains a reproducible experiment that measures how three activation functions (`relu`, `sigmoid`, `tanh`) and three optimizers (`sgd`, `adam`, `rmsprop`) affect training and test performance on the MNIST digit classification task using a small fully-connected neural network (MLP).

The code trains every combination of (optimizer, activation) for 10 epochs, saves each trained model, writes a results CSV, and generates a few plots (bar chart, heatmap, confusion matrix for the best model).

---

## Real-World Use Cases

* Bank check processing.
* Postal code recognition (postal sorting).
* Digitizing handwritten forms (OCR).
* Mobile document / note scanning apps.

---

## Tech Stack

* Python 3.8+
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib, Seaborn
* scikit-learn (confusion matrix)
* tqdm (progress bars)

Install dependencies:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn tqdm
```

---

## Project Structure

```
MNIST-Activations-Optimizers/
│── train.py               
│── models/                
│── results/             
│   ├── results.csv
│   ├── results.png
│   ├── results_heatmap.png
│   └── confusion_matrix.png
└── README.md
```

---

## About the Provided Script

The version of the script associated with this README does the following (matches the provided code):

* Uses the built-in `mnist` dataset from `tensorflow.keras.datasets`.
* Normalizes input images to `[0, 1]` and keeps the original `(28, 28)` shape until the model `Flatten` layer.
* Builds a small MLP with this structure:

  * `Flatten(input_shape=(28, 28))`
  * `Dense(128, activation=<activation>)`
  * `Dense(64, activation=<activation>)`
  * `Dense(10, activation="softmax")`
* Trains for **10 epochs** with `batch_size=128` and `validation_split=0.1` for each of the 9 combinations.
* Logs training progress (Keras output) and displays a `tqdm` progress bar for the high-level loop.
* Saves each model to `models/model_<optimizer>_<activation>.h5`.
* Tracks the best model by **test accuracy** and uses it to create a confusion matrix saved to `results/confusion_matrix.png`.

### Hyperparameters (as used in the script)

* Activations: `['relu', 'sigmoid', 'tanh']`
* Optimizers: `['sgd', 'adam', 'rmsprop']` (passed as strings to `model.compile`)
* Epochs: `10`
* Batch size: `128`
* Validation split: `0.1`

> Note: The script uses each optimizer’s default settings when you pass the string (e.g. default learning rates). If you want reproducible results or specific learning rates, create optimizer objects (e.g. `optimizers.SGD(learning_rate=0.01)`) and set seeds for NumPy/TensorFlow.

---

## Running the Experiment

From the project root, run:

```bash
python train.py
```

What will happen:

* The script trains 9 models (3 activations × 3 optimizers).
* Trained models are saved in `models/` as `model_<optimizer>_<activation>.h5`.
* A results table `results/results.csv` is written.
* Plots are saved in `results/`:

  * `results.png` — Bar chart of Test Accuracy by Optimizer & Activation.
  * `results_heatmap.png` — Pivot heatmap (Optimizer vs Activation) of Test Accuracy.
  * `confusion_matrix.png` — Confusion matrix for the best-performing model.
* The script prints a final sorted results table and identifies the best model.

---

## Results Format

`results/results.csv` contains the following columns (one row per experiment):

\| Optimizer | Activation | Train Accuracy | Test Accuracy | Train Loss | Test Loss |

Plots and the CSV let you compare which optimizer + activation pairs generalize best on MNIST with the provided small MLP.

---

## Example Output (format)

At the end of training you will see a printed summary similar to:

```
Final Results Summary:
Optimizer Activation  Train Accuracy  Test Accuracy  Train Loss  Test Loss
   adam       relu          0.9871         0.9827       0.0456     0.0623
 rmsprop      tanh          0.9763         0.9731       0.0652     0.0754
    sgd       relu          0.9345         0.9204       0.2312     0.2546
```

(Actual numbers will vary by environment, TF version, and random initialization.)

---
