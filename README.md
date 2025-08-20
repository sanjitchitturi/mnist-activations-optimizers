# MNIST Activations & Optimizers

A study of how **activation functions** (ReLU, Sigmoid, Tanh) and **optimizers** (SGD, Adam, RMSProp) affect performance on the **MNIST handwritten digit classification** task.

---

## Real-World Use Case

Digit recognition is widely used in:
- Bank check processing
- Postal code recognition
- Digitizing handwritten forms (OCR)

---

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, scikit-learn

---

## Results

The script will train multiple models and save:
- `results.csv` → metrics table  
- `results.png` → bar chart of test accuracy  
- `results_heatmap.png` → accuracy heatmap (optimizer vs activation)  
- `confusion_matrix.png` → confusion matrix for best model  
- `models/` → saved models  

Example output (after running):

![Results](results.png)

---

## How to Run
```bash
git clone https://github.com/sanjitchitturi/MNIST-Activations-Optimizers.git
cd MNIST-Activations-Optimizers
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_experiments.py
