"""
infer_student_bn.py

Script που φορτώνει το Student Bayesian Network από αρχείο .joblib
και εκτελεί μερικά βασικά ερωτήματα inference με VariableElimination.

Τα παραδείγματα περιλαμβάνουν:

- P(Grade | Difficulty=hard, Intelligence=high)
- P(Intelligence | Grade=A, SAT=high)
- P(Intelligence | Letter=strong)
"""

from pathlib import Path

import joblib
from pgmpy.inference import VariableElimination


# Φάκελος όπου αποθηκεύεται το μοντέλο .joblib
MODELS_DIR = Path(__file__).resolve().parent / "models"


def load_model(model_name: str = "student_bn.joblib"):
    """
    Φορτώνει το αποθηκευμένο Student Bayesian Network.

    Παράμετροι
    ----------
    model_name : str
        Όνομα του .joblib αρχείου μέσα στον φάκελο models/.
    """
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκε το αρχείο {model_path}.\n"
            "Τρέξε πρώτα το train_student_bn.py για να δημιουργήσεις το δίκτυο."
        )
    return joblib.load(model_path)


def run_examples():
    """
    Εκτελεί μερικά ενδεικτικά ερωτήματα inference στο Student BN.

    Χρησιμοποιούμε τον αλγόριθμο Variable Elimination όπως υλοποιείται
    στη βιβλιοθήκη pgmpy.
    """
    model = load_model()
    infer = VariableElimination(model)

    print("=== Παράδειγμα 1: P(Grade | Difficulty=hard, Intelligence=high) ===")
    q1 = infer.query(
        variables=["Grade"],
        evidence={"Difficulty": "hard", "Intelligence": "high"},
        show_progress=False,
    )
    print(q1)

    print("\n=== Παράδειγμα 2: P(Intelligence | Grade=A, SAT=high) ===")
    q2 = infer.query(
        variables=["Intelligence"],
        evidence={"Grade": "A", "SAT": "high"},
        show_progress=False,
    )
    print(q2)

    print("\n=== Παράδειγμα 3: P(Intelligence | Letter=strong) ===")
    q3 = infer.query(
        variables=["Intelligence"],
        evidence={"Letter": "strong"},
        show_progress=False,
    )
    print(q3)


if __name__ == "__main__":
    run_examples()
