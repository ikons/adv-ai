"""
train_gaussian_nb_iris.py

Παράδειγμα χρήσης Gaussian Naive Bayes στο Iris dataset.
Το dataset είναι ενσωματωμένο στη scikit-learn και περιέχει
συνεχή χαρακτηριστικά, ιδανικά για Gaussian NB.

Το script αυτό:

- φορτώνει το Iris dataset,
- κανονικοποιεί τα χαρακτηριστικά με StandardScaler,
- χωρίζει τα δεδομένα σε train / validation,
- εκπαιδεύει GaussianNB,
- εκτυπώνει αναλυτικό classification_report,
- αποθηκεύει το μοντέλο και τον scaler σε αρχεία .joblib,
- αποθηκεύει ένα διάγραμμα confusion matrix σε .png.

Έχει πολλά σχόλια στα ελληνικά ώστε να είναι κατάλληλο
για διδακτικούς σκοπούς.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # backend χωρίς οθόνη, κατάλληλο για scripts
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


# Φάκελος αποθήκευσης μοντέλων / διαγραμμάτων
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train(test_size: float = 0.2, random_state: int = 0):
    """
    Εκπαίδευση Gaussian Naive Bayes στο Iris dataset.

    Παράμετροι
    ----------
    test_size : float
        Ποσοστό των δεδομένων που χρησιμοποιείται για validation (0–1).
    random_state : int
        Σπόρος τυχαιότητας για αναπαραγωγιμότητα.
    """
    # ---------------------------------------------------------
    # 1. Φόρτωση του Iris dataset
    # ---------------------------------------------------------
    iris = load_iris()
    X = iris.data          # continuous χαρακτηριστικά: sepal/petal length/width
    y = iris.target        # 0, 1, 2
    class_names = iris.target_names

    print("Σχήμα X:", X.shape)
    print("Κλάσεις:", class_names)

    # ---------------------------------------------------------
    # 2. Προαιρετική κανονικοποίηση με StandardScaler
    # ---------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------------------------
    # 3. train / validation split
    # ---------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        stratify=y,          # διατηρούμε την αναλογία κλάσεων
        random_state=random_state,
    )

    # ---------------------------------------------------------
    # 4. Ορισμός και εκπαίδευση GaussianNB
    # ---------------------------------------------------------
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 5. Αξιολόγηση στο validation set
    # ---------------------------------------------------------
    y_pred = gnb.predict(X_val)

    print("=== Gaussian Naive Bayes στο Iris dataset ===")
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=class_names,
            digits=3,
        )
    )

    # ---------------------------------------------------------
    # 6. Αποθήκευση μοντέλου και scaler
    # ---------------------------------------------------------
    model_path = MODELS_DIR / "gaussian_nb_iris.joblib"
    scaler_path = MODELS_DIR / "gaussian_nb_iris_scaler.joblib"

    joblib.dump(gnb, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n✅ Αποθηκεύτηκε το μοντέλο στο: {model_path}")
    print(f"✅ Αποθηκεύτηκε ο scaler στο: {scaler_path}")

    # ---------------------------------------------------------
    # 7. Confusion matrix σε .png
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_val,
        y_pred,
        display_labels=class_names,
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion matrix – Gaussian NB (Iris)")
    fig.tight_layout()

    cm_path = MODELS_DIR / "gaussian_nb_iris_cm.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"📊 Αποθηκεύτηκε το confusion matrix στο: {cm_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Εκπαίδευση Gaussian Naive Bayes στο Iris dataset."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Ποσοστό validation set (0–1).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Σπόρος τυχαιότητας για αναπαραγωγιμότητα.",
    )

    args = parser.parse_args()

    train(
        test_size=args.test_size,
        random_state=args.random_state,
    )
