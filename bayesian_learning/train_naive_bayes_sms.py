"""
train_naive_bayes_sms.py

Script εκπαίδευσης μοντέλου Multinomial Naive Bayes για ταξινόμηση
SMS μηνυμάτων σε "ham" (κανονικά) και "spam" (ανεπιθύμητα).

Το script είναι γραμμένο με πολλά σχόλια στα ελληνικά ώστε να είναι
κατάλληλο για φοιτητές που ξεκινούν τώρα με Machine Learning.

Η βασική ιδέα του Naive Bayes:

- Χρησιμοποιούμε τον τύπο του Bayes:
  P(y | x) ∝ P(x | y) P(y)
- Υποθέτουμε ότι τα χαρακτηριστικά x_i είναι ανεξάρτητα μεταξύ τους
  δεδομένης της κλάσης y (υπόθεση "naive").
- Για κείμενο, το Multinomial Naive Bayes μοντελοποιεί τις συχνότητες
  εμφάνισης των λέξεων μέσα σε κάθε κλάση.

Εδώ, τα χαρακτηριστικά προκύπτουν από TfidfVectorizer πάνω στα κείμενα
των SMS, και το μοντέλο είναι MultinomialNB (της scikit-learn).
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Χρησιμοποιούμε backend χωρίς οθόνη για αποθήκευση εικόνων
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# -------------------------------------------------------------
# Ορισμός βασικών paths
# -------------------------------------------------------------

# Ριζικός φάκελος του repo (θεωρούμε ότι ο φάκελος bayesian_learning
# βρίσκεται απευθείας κάτω από τη ρίζα).
ROOT = Path(__file__).resolve().parents[1]

# Αρχείο δεδομένων (CSV) που πρέπει να υπάρχει στον φάκελο data/
DATA_PATH = ROOT / "data" / "sms_spam.csv"

# Φάκελος για αποθήκευση μοντέλων και διαγραμμάτων
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_sms_spam():
    """
    Φορτώνει το SMS Spam dataset από το data/sms_spam.csv.

    Αναμένεται CSV με στήλες:
    - 'label': τιμή 'ham' ή 'spam'
    - 'text' : το κείμενο του SMS (string)

    Επιστρέφει:
    - X: pandas Series με τα κείμενα
    - y: pandas Series με τις ετικέτες σε μορφή 0/1
      (0 = ham, 1 = spam)
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκε το αρχείο δεδομένων: {DATA_PATH}\n"
            "Βεβαιώσου ότι έχεις αποθηκεύσει το SMS Spam dataset ως "
            "'sms_spam.csv' στον φάκελο data/."
        )

    # Φορτώνουμε με encoding='latin-1' γιατί το sms_spam.csv δεν είναι UTF-8
    df = pd.read_csv(DATA_PATH, encoding='latin-1')

    # Το αρχικό CSV έχει στήλες v1 (label) και v2 (text)
    # Κρατάμε μόνο αυτές τις δύο και μετονομάζουμε για ευκολία
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'text']

    # Αφαιρούμε τυχόν γραμμές με κενά στις βασικές στήλες
    df = df.dropna(subset=['label', 'text']).copy()

    # Κανονικοποίηση labels (strip και lowercase)
    df['label'] = df['label'].str.strip().str.lower()
    # Κρατάμε μόνο γραμμές με 'ham' ή 'spam'
    df = df[df['label'].isin(['ham', 'spam'])].copy()

    # Χαρτογράφηση των labels σε 0/1
    # ham -> 0 (κανονικό μήνυμα)
    # spam -> 1 (ανεπιθύμητο)
    y = (df["label"] == "spam").astype(int)

    # Τα κείμενα των μηνυμάτων
    X = df["text"].astype(str)

    return X, y


def train(alpha: float = 1.0, max_features: int = 10000, test_size: float = 0.2, random_state: int = 0):
    """
    Εκπαιδεύει ένα μοντέλο Multinomial Naive Bayes για ταξινόμηση SMS.

    Παράμετροι
    ----------
    alpha : float
        Παράμετρος εξομάλυνσης Laplace (MultinomialNB.alpha).
        Τιμές μεγαλύτερες του 0 αποτρέπουν μηδενικές πιθανότητες.

    max_features : int
        Μέγιστος αριθμός χαρακτηριστικών TF-IDF που θα κρατήσουμε.
        Αν το dataset έχει πάρα πολλές διαφορετικές λέξεις,
        περιορίζουμε το λεξιλόγιο σε max_features όρους.

    test_size : float
        Ποσοστό των δεδομένων που θα χρησιμοποιηθούν για validation
        (π.χ. 0.2 = 20% του dataset).

    random_state : int
        Σπόρος τυχαιότητας για αναπαραγωγιμότητα των αποτελεσμάτων.
    """
    # ---------------------------------------------------------
    # 1. Φόρτωση δεδομένων
    # ---------------------------------------------------------
    X, y = load_sms_spam()

    # Χωρισμός σε train / validation με stratify ώστε να διατηρήσουμε
    # την αναλογία ham / spam και στα δύο σύνολα.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # ---------------------------------------------------------
    # 2. Ορισμός Pipeline (TF-IDF Vectorizer + MultinomialNB)
    # ---------------------------------------------------------
    # Το Pipeline εφαρμόζει διαδοχικά μετασχηματιστές και ταξινομητή.
    # Εδώ:
    # - πρώτα μετατρέπει τα κείμενα σε TF-IDF features,
    # - μετά εφαρμόζει τον Multinomial Naive Bayes.
    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    stop_words="english",  # το dataset είναι συνήθως στα αγγλικά
                ),
            ),
            ("nb", MultinomialNB(alpha=alpha)),
        ]
    )

    # ---------------------------------------------------------
    # 3. Εκπαίδευση του μοντέλου
    # ---------------------------------------------------------
    pipe.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 4. Αξιολόγηση στο validation set
    # ---------------------------------------------------------
    y_pred = pipe.predict(X_val)

    print("=== Multinomial Naive Bayes στο SMS Spam dataset ===")
    print(f"Δείγματα train: {len(X_train)}, validation: {len(X_val)}")
    print()
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=["ham", "spam"],
            digits=3,
        )
    )

    # ---------------------------------------------------------
    # 5. Αποθήκευση εκπαιδευμένου pipeline (vectorizer + μοντέλο)
    # ---------------------------------------------------------
    model_file = MODELS_DIR / f"naive_bayes_sms_alpha{alpha}.joblib"
    joblib.dump(pipe, model_file)
    print(f"\n✅ Αποθηκεύτηκε το μοντέλο στο: {model_file}")

    # ---------------------------------------------------------
    # 6. Ερμηνεία: priors και πιο χαρακτηριστικές λέξεις
    # ---------------------------------------------------------
    nb = pipe.named_steps["nb"]
    tfidf = pipe.named_steps["tfidf"]

    print("\nLog-priors κλάσεων (P(class)) όπως τα μαθαίνει το μοντέλο:")
    for cls, logp in zip(["ham", "spam"], nb.class_log_prior_):
        print(f"  {cls:>4}: {logp: .3f}")

    # Οι πιθανότητες των χαρακτηριστικών για κάθε κλάση
    feature_names = np.array(tfidf.get_feature_names_out())

    # Δείχνουμε τις πιο "χαρακτηριστικές" λέξεις για spam και ham
    spam_top_idx = nb.feature_log_prob_[1].argsort()[-15:][::-1]
    ham_top_idx = nb.feature_log_prob_[0].argsort()[-15:][::-1]

    print("\nTop-15 λέξεις με τη μεγαλύτερη πιθανότητα στην κλάση 'spam':")
    print(", ".join(feature_names[spam_top_idx]))

    print("\nTop-15 λέξεις με τη μεγαλύτερη πιθανότητα στην κλάση 'ham':")
    print(", ".join(feature_names[ham_top_idx]))

    # ---------------------------------------------------------
    # 7. Confusion matrix στο validation set
    # ---------------------------------------------------------
    cm = confusion_matrix(y_val, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
    disp.plot(ax=ax)
    ax.set_title("Confusion matrix (validation set)")
    fig.tight_layout()

    fig_path = MODELS_DIR / f"naive_bayes_sms_alpha{alpha}_cm.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"📊 Αποθηκεύτηκε το διάγραμμα confusion matrix στο: {fig_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Εκπαίδευση Multinomial Naive Bayes στο SMS Spam dataset."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Παράμετρος εξομάλυνσης Laplace (MultinomialNB.alpha).",

    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=10000,
        help="Μέγιστος αριθμός χαρακτηριστικών TF-IDF.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Μέγεθος validation set ως ποσοστό (0–1).",

    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Σπόρος τυχαιότητας για αναπαραγωγιμότητα.",
    )

    args = parser.parse_args()

    train(
        alpha=args.alpha,
        max_features=args.max_features,
        test_size=args.test_size,
        random_state=args.random_state,
    )
