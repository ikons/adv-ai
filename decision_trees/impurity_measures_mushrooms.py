from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------
# Ορισμός διαδρομής προς το dataset "mushrooms.csv"
# ----------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "mushrooms.csv"


# ----------------------------------------------
# ΥΠΟΛΟΓΙΣΜΟΣ ENTROPY
# ----------------------------------------------
def entropy(y: pd.Series) -> float:
    """
    Υπολογίζει την εντροπία (entropy) για μια στήλη κατηγορικών τιμών.
    Η εντροπία μετράει την "αβεβαιότητα" μιας κατανομής.

    Τύπος:
        H = - Σ (p_i * log2(p_i))

    όπου p_i είναι η σχετική συχνότητα κάθε κατηγορίας.
    Αν όλες οι τιμές είναι ίδιες -> H = 0 (τέλεια καθαρότητα)
    Αν οι τιμές είναι ισοκατανεμημένες -> H = log2(k) (μέγιστη αβεβαιότητα)
    """
    p = y.value_counts(normalize=True)  # ποσοστό κάθε κατηγορίας
    return float(-(p * np.log2(p)).sum())


# ----------------------------------------------
# ΥΠΟΛΟΓΙΣΜΟΣ GINI IMPURITY
# ----------------------------------------------
def gini_impurity(y: pd.Series) -> float:
    """
    Υπολογίζει την αβεβαιότητα Gini για μια στήλη κατηγορικών τιμών.

    Τύπος:
        G = 1 - Σ (p_i)^2

    Όπου p_i η πιθανότητα κάθε κατηγορίας.
    Όσο μικρότερη η G, τόσο πιο "καθαρός" ο κόμβος.
    """
    p = y.value_counts(normalize=True)
    return float(1.0 - (p**2).sum())


# ----------------------------------------------
# ΥΠΟΛΟΓΙΣΜΟΣ ΠΛΗΡΟΦΟΡΙΑΣ ΚΑΙ ΔΕΙΚΤΩΝ ΓΙΑ ΕΝΑ ΧΑΡΑΚΤΗΡΙΣΤΙΚΟ
# ----------------------------------------------
def feature_scores(df: pd.DataFrame, feature: str, target: str):
    """
    Υπολογίζει 4 δείκτες καθαρότητας για ένα κατηγορικό χαρακτηριστικό:
      - Information Gain
      - Split Information
      - Gain Ratio
      - Gini Gain

    Οι δείκτες αυτοί δείχνουν πόσο "πληροφοριακό" είναι το χαρακτηριστικό
    για την πρόβλεψη της μεταβλητής-στόχου.
    """

    # Στόχος (target variable)
    y = df[target]
    # Χαρακτηριστικό (feature)
    X = df[feature]

    n = len(df)  # πλήθος δειγμάτων

    # Υπολογισμός εντροπίας και Gini impurity πριν το split
    H_before = entropy(y)
    G_before = gini_impurity(y)

    # Αρχικοποίηση μεταβλητών για "μετά το split"
    H_after = 0.0
    G_after = 0.0
    split_info = 0.0

    # Για κάθε τιμή του χαρακτηριστικού (π.χ. cap-color = brown, white, red, ...)
    for v, subset in df.groupby(feature):
        weight = len(subset) / n  # σχετικό βάρος του υποσυνόλου
        y_sub = subset[target]    # στόχος στο υποσύνολο

        # Υπολογισμός εντροπίας και Gini στο υποσύνολο
        H_after += weight * entropy(y_sub)
        G_after += weight * gini_impurity(y_sub)

        # Split Information — μετράει το "κόστος" του διαχωρισμού
        # (προτιμά χαρακτηριστικά με λιγότερες διακριτές τιμές)
        if weight > 0:
            split_info -= weight * np.log2(weight)

    # Υπολογισμός δεικτών
    info_gain = H_before - H_after       # Information Gain
    gini_gain = G_before - G_after       # Gini Gain
    gain_ratio = info_gain / split_info if split_info > 0 else 0.0  # Gain Ratio

    return info_gain, split_info, gain_ratio, gini_gain


# ----------------------------------------------
# ΚΥΡΙΑ ΣΥΝΑΡΤΗΣΗ ΕΚΤΕΛΕΣΗΣ
# ----------------------------------------------
def main():
    # Φόρτωση του dataset Mushroom (από UCI)
    df = pd.read_csv(DATA_PATH)

    target = "class"  # τιμή 'e' (edible) ή 'p' (poisonous)
    features = [c for c in df.columns if c != target]

    rows = []

    # Υπολογισμός μετρικών για κάθε χαρακτηριστικό
    for feat in features:
        ig, si, gr, gg = feature_scores(df, feat, target)
        rows.append(
            {
                "feature": feat,
                "num_values": df[feat].nunique(),
                "info_gain": ig,
                "split_info": si,
                "gain_ratio": gr,
                "gini_gain": gg,
            }
        )

    # Δημιουργία πίνακα αποτελεσμάτων
    scores = pd.DataFrame(rows)

    # Εμφάνιση κορυφαίων χαρακτηριστικών με βάση κάθε μέτρο
    print("\n=== Top features by Information Gain ===")
    print(scores.sort_values("info_gain", ascending=False).head(10))

    print("\n=== Top features by Gain Ratio ===")
    print(scores.sort_values("gain_ratio", ascending=False).head(10))

    print("\n=== Top features by Gini Gain ===")
    print(scores.sort_values("gini_gain", ascending=False).head(10))


# ----------------------------------------------
# ΕΚΤΕΛΕΣΗ SCRIPT
# ----------------------------------------------
if __name__ == "__main__":
    main()
