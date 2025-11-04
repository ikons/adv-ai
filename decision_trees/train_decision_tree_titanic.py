from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ----------------------------------------------
# ΡΥΘΜΙΣΕΙΣ ΔΙΑΔΡΟΜΩΝ ΚΑΙ ΦΑΚΕΛΩΝ
# ----------------------------------------------

# Ο φάκελος-ρίζα του project (ένας κατάλογος πάνω από το τρέχον αρχείο)
ROOT = Path(__file__).resolve().parents[1]

# Διαδρομή προς το αρχείο δεδομένων Titanic (π.χ. "data/titanic_train.csv")
DATA_PATH = ROOT / "data" / "titanic_train.csv"

# Δημιουργούμε φάκελο όπου θα αποθηκεύονται τα εκπαιδευμένα μοντέλα και τα διαγράμματα
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ----------------------------------------------
# ΦΟΡΤΩΣΗ ΚΑΙ ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ ΔΕΔΟΜΕΝΩΝ
# ----------------------------------------------

def load_titanic():
    """
    Φόρτωση του dataset Titanic και προετοιμασία των δεδομένων.

    Επιστρέφει:
    - X: τα χαρακτηριστικά εισόδου (features)
    - y: τις ετικέτες στόχου (αν επέζησε ή όχι)
    - preprocessor: ColumnTransformer που κάνει One-Hot Encoding στις κατηγορικές στήλες
    - cat_cols: λίστα κατηγορικών πεδίων
    - num_cols: λίστα αριθμητικών πεδίων
    """
    df = pd.read_csv(DATA_PATH)

    # Επιλογή των πεδίων που μας ενδιαφέρουν
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    target = "Survived"

    df = df[features + [target]]

    # Αντιμετώπιση ελλειπτικών τιμών
    df["Age"] = df["Age"].fillna(df["Age"].median())         # μέση τιμή ηλικίας (διάμεσος)
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # πιο συχνή τιμή

    X = df[features]
    y = df[target]

    # Καθορίζουμε ποιες στήλες είναι κατηγορικές και ποιες αριθμητικές
    cat_cols = ["Sex", "Embarked"]  # κατηγορικές μεταβλητές (τιμές τύπου string)
    num_cols = [c for c in features if c not in cat_cols]  # αριθμητικές (π.χ. Age, Fare κ.λπ.)

    # Ο ColumnTransformer εφαρμόζει διαφορετική μετατροπή ανά ομάδα στηλών
    # Εδώ:
    #  - Στις κατηγορικές (π.χ. "Sex" και "Embarked") εφαρμόζεται One-Hot Encoding.
    #  - Στις αριθμητικές (π.χ. "Age", "Fare") δεν εφαρμόζεται καμία αλλαγή (passthrough).
    #
    # ➤ Τι κάνει ο OneHotEncoder:
    #    Ο OneHotEncoder μετατρέπει τις κατηγορικές τιμές (π.χ. "male", "female")
    #    σε ξεχωριστές δυαδικές στήλες (0 ή 1), μία για κάθε κατηγορία.
    #    Παράδειγμα:
    #       Στήλη: Sex
    #          male
    #          female
    #       Μετά την κωδικοποίηση:
    #          Sex_male | Sex_female
    #          1         | 0
    #          0         | 1
    #
    #    Το ίδιο γίνεται και για το πεδίο "Embarked":
    #       C, Q, S  ->  Embarked_C | Embarked_Q | Embarked_S
    #
    #    Έτσι, όλες οι κατηγορικές τιμές μετατρέπονται σε αριθμητική μορφή (0/1),
    #    ώστε να μπορούν να εισαχθούν στο μοντέλο μηχανικής μάθησης.
    #
    #    Το όρισμα handle_unknown="ignore" λέει στο encoder να αγνοεί
    #    άγνωστες κατηγορίες που ενδέχεται να εμφανιστούν κατά την πρόβλεψη.
    #
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    return X, y, preprocessor, cat_cols, num_cols


# ----------------------------------------------
# ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΟΥ
# ----------------------------------------------

def train(criterion="gini", max_depth=None, random_state=0, test_size=0.2):
    """
    Εκπαίδευση δέντρου αποφάσεων (DecisionTreeClassifier) στο Titanic dataset.

    Παράμετροι:
    - criterion: μέτρο ακαθαρσίας ('gini' ή 'entropy')
    - max_depth: μέγιστο βάθος δέντρου (None = χωρίς περιορισμό)
    - random_state: σπόρος τυχαιότητας για αναπαραγωγιμότητα
    - test_size: ποσοστό δεδομένων για validation
    """
    X, y, preprocessor, cat_cols, num_cols = load_titanic()

    # Διαχωρισμός σε train και validation σύνολο
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Ορισμός του ταξινομητή
    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state,
    )

    # Δημιουργία pipeline: πρώτα προεπεξεργασία, μετά εκπαίδευση του μοντέλου
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

    # Εκπαίδευση
    pipe.fit(X_train, y_train)

    # Πρόβλεψη και αξιολόγηση
    y_pred = pipe.predict(X_val)
    print(f"=== Decision Tree (criterion={criterion}, max_depth={max_depth}) ===")
    print(classification_report(y_val, y_pred, digits=3))

    # Αποθήκευση του μοντέλου (περιλαμβάνει και τον preprocessor)
    model_file = (
        MODELS_DIR
        / f"decision_tree_titanic_{criterion}_depth{max_depth or 'None'}.joblib"
    )
    joblib.dump(pipe, model_file)
    print(f"Saved model to {model_file}")

    # Αν το δέντρο είναι ρηχό, το σχεδιάζουμε σε εικόνα για οπτική κατανόηση
    if max_depth is not None and max_depth <= 4:
        tree_clf = pipe.named_steps["model"]
        ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names = cat_feature_names + num_cols

        plt.figure(figsize=(22, 10))
        plot_tree(
            tree_clf,
            feature_names=feature_names,
            class_names=["Died", "Survived"],
            filled=True,
            rounded=True,
            fontsize=8,
        )
        fig_path = (
            MODELS_DIR
            / f"decision_tree_titanic_{criterion}_depth{max_depth}.png"
        )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Saved tree plot to {fig_path}")


# ----------------------------------------------
# ΚΥΡΙΟ ΜΠΛΟΚ ΕΚΤΕΛΕΣΗΣ (CLI)
# ----------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a decision tree on Titanic data.")
    parser.add_argument(
        "--criterion",
        choices=["gini", "entropy"],
        default="gini",
        help="Μέτρο ακαθαρσίας κόμβων (gini ή entropy).",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Μέγιστο βάθος δέντρου (None = χωρίς όριο).",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Μέγεθος validation set ως ποσοστό (π.χ. 0.2 = 20%).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Σπόρος τυχαιότητας για επαναληψιμότητα.",
    )

    args = parser.parse_args()

    train(
        criterion=args.criterion,
        max_depth=args.max_depth,
        random_state=args.random_state,
        test_size=args.test_size,
    )