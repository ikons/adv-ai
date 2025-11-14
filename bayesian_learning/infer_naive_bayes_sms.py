"""
infer_naive_bayes_sms.py

Script για inference (πρόβλεψη) με το εκπαιδευμένο μοντέλο
Multinomial Naive Bayes πάνω σε νέα SMS μηνύματα.

Η βασική λογική:

1. Φορτώνουμε το αποθηκευμένο pipeline (TF-IDF vectorizer + Naive Bayes)
   από το φάκελο models/.
2. Δημιουργούμε μια λίστα με νέα μηνύματα (strings).
3. Καλούμε pipe.predict και pipe.predict_proba για να πάρουμε
   την προβλεπόμενη κλάση και τις αντίστοιχες πιθανότητες.
"""

from pathlib import Path

import joblib
import pandas as pd


# Φάκελος όπου αποθηκεύονται τα μοντέλα (.joblib)
MODELS_DIR = Path(__file__).resolve().parent / "models"


def load_model(model_name=None):
    """
    Φορτώνει το εκπαιδευμένο pipeline Naive Bayes (joblib).

    Αν δεν δοθεί συγκεκριμένο όνομα αρχείου, χρησιμοποιούμε το default:
    'naive_bayes_sms_alpha1.0.joblib'.
    """
    if model_name is None:
        model_name = "naive_bayes_sms_alpha1.0.joblib"

    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκε το μοντέλο {model_path}.\n"
            "Εκτέλεσε πρώτα το train_naive_bayes_sms.py για να εκπαιδεύσεις το μοντέλο."
        )

    return joblib.load(model_path)


def predict_messages(messages, model_name=None):
    """
    Κάνει πρόβλεψη ham/spam για μια λίστα από μηνύματα SMS.

    Παράμετροι
    ----------
    messages : list[str]
        Λίστα με strings (τα μηνύματα προς ταξινόμηση).

    model_name : str ή None
        Όνομα αρχείου .joblib στο φάκελο models/.
        Αν είναι None, χρησιμοποιείται το default.
    """
    pipe = load_model(model_name)

    # Μετατρέπουμε τη λίστα σε pandas Series ώστε να είναι συνεπής με το training
    X_new = pd.Series(messages)
    y_pred = pipe.predict(X_new)
    y_proba = pipe.predict_proba(X_new)

    # Εκτύπωση αποτελεσμάτων με αναλυτικές πληροφορίες
    for text, label, proba in zip(messages, y_pred, y_proba):
        cls = "spam" if label == 1 else "ham"
        p_ham, p_spam = proba[0], proba[1]

        print("-" * 72)
        print("Μήνυμα:")
        print(text)
        print()
        print(f"→ Πρόβλεψη: {cls.upper()}")
        print(f"   P(ham | x)  = {p_ham:.3f}")
        print(f"   P(spam | x) = {p_spam:.3f}")


if __name__ == "__main__":
    # Μερικά ενδεικτικά παραδείγματα (αγγλικά SMS όπως στο κλασικό dataset)
    example_messages = [
        "WINNER!! You have won a 1000$ cash prize. Call now to claim.",
        "Hey, are we still on for coffee tomorrow?",
        "FREE entry in 2 a weekly competition to win FA Cup final tickets. Text WIN to 87121 now!",
        "I'll be there in 10 minutes.",
    ]

    predict_messages(example_messages)
