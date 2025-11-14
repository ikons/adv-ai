"""
infer_gaussian_nb_iris.py

Script για inference (πρόβλεψη) με το εκπαιδευμένο Gaussian Naive Bayes
στο Iris dataset.

Φορτώνει:

- τον κανονικοποιητή χαρακτηριστικών (StandardScaler),
- το εκπαιδευμένο μοντέλο GaussianNB,

και στη συνέχεια υπολογίζει προβλέψεις για μερικά ενδεικτικά
4-διάστατα διανύσματα χαρακτηριστικών [sepal len, sepal wid, petal len, petal wid].
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris


MODELS_DIR = Path(__file__).resolve().parent / "models"


def load_model_and_scaler(
    model_name: str = "gaussian_nb_iris.joblib",
    scaler_name: str = "gaussian_nb_iris_scaler.joblib",
):
    """
    Φορτώνει το GaussianNB μοντέλο και τον StandardScaler από τον φάκελο models/.

    Αν δεν βρεθούν τα αρχεία, ενημερώνει τον χρήστη να τρέξει πρώτα
    το train_gaussian_nb_iris.py.
    """
    model_path = MODELS_DIR / model_name
    scaler_path = MODELS_DIR / scaler_name

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκαν τα αρχεία:\n"
            f"  - {model_path}\n"
            f"  - {scaler_path}\n"
            "Τρέξε πρώτα: python -m bayesian_learning.train_gaussian_nb_iris"
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def demo_inference():
    """
    Εκτελεί inference σε μερικά ενδεικτικά δείγματα του Iris dataset:

    - παίρνουμε μερικές πραγματικές εγγραφές από το dataset,
    - τις κανονικοποιούμε με τον ίδιο scaler,
    - εκτυπώνουμε κλάση, όνομα κλάσης και posterior πιθανότητες.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    model, scaler = load_model_and_scaler()

    # Παίρνουμε μερικά δείγματα: το πρώτο, το 50ό και το 100ό
    indices = [0, 50, 100]
    X_samples = X[indices]
    y_true = y[indices]

    # Κανονικοποίηση με τον ίδιο scaler που χρησιμοποιήθηκε στο training
    X_scaled = scaler.transform(X_samples)

    # Πρόβλεψη κλάσης και πιθανοτήτων
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)

    for i, (x, true_label, pred_label, proba) in enumerate(
        zip(X_samples, y_true, y_pred, y_proba)
    ):
        print("-" * 72)
        print(f"Δείγμα #{indices[i]}")
        print(f"Χαρακτηριστικά (sepal_len, sepal_wid, petal_len, petal_wid): {x}")
        print(f"Πραγματική κλάση: {target_names[true_label]}")
        print(f"Πρόβλεψη       : {target_names[pred_label]}")
        print("Posterior πιθανότητες:")
        for cls_idx, cls_name in enumerate(target_names):
            print(f"  P({cls_name} | x) = {proba[cls_idx]:.3f}")


if __name__ == "__main__":
    demo_inference()
