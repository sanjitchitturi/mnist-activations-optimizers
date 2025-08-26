# MNIST Activations & Optimizers

A study of how different **activation functions** (ReLU, Sigmoid, Tanh) and **optimizers** (SGD, Adam, RMSProp) affect performance on the **MNIST handwritten digit classification** task.

---

## Real-World Use Cases

Digit recognition is widely used in:

* Bank check processing
* Postal code recognition (postal sorting machines)
* Digitizing handwritten forms (OCR)
* Mobile apps for note scanning

---

## Tech Stack

* **Python 3.8+**
* **TensorFlow / Keras** (model training)
* **NumPy, Pandas** (data processing)
* **Matplotlib, Seaborn** (visualizations)
* **scikit-learn** (confusion matrix)
* **tqdm** (progress bars)

Install dependencies:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn tqdm
```

---

## Project Structure

```
mnist_activations_optimizers/
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

## Running the Project

Run the experiment:

```bash
python train.py
```

This will:

* Train **9 models** (3 activations × 3 optimizers)
* Save trained models in `models/`
* Save metrics + plots in `results/`
* Print a **summary table** at the end with sorted results

---

## Results Generated

1. **results.csv** → metrics table:
   \| Optimizer | Activation | Train Accuracy | Test Accuracy | Train Loss | Test Loss |

2. **results.png** → bar chart of test accuracy.

3. **results\_heatmap.png** → heatmap (optimizer vs activation).

4. **confusion\_matrix.png** → confusion matrix of the best model.

---

## Example Output

At the end of training you’ll see a summary table like:

```
Final Results Summary:
Optimizer Activation  Train Accuracy  Test Accuracy  Train Loss  Test Loss
     adam       relu          0.9956         0.9842     0.0123     0.0641
   rmsprop       relu          0.9947         0.9825     0.0157     0.0703
     sgd       tanh          0.9502         0.9251     0.1953     0.2404
```

---

## Future Improvements

* Try other activations (LeakyReLU, ELU, GELU)
* Add dropout or batch normalization
* Use CNN layers for higher accuracy
* Hyperparameter tuning (learning rate, batch size)

---
