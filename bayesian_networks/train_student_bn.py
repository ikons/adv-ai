"""
train_student_bn.py

Script που δημιουργεί το κλασικό Student Bayesian Network
με τη βιβλιοθήκη pgmpy και το αποθηκεύει σε αρχείο .joblib.

Το δίκτυο περιλαμβάνει τις μεταβλητές:

- Difficulty   ∈ {easy, hard}
- Intelligence ∈ {low, high}
- Grade        ∈ {A, B, C}
- SAT          ∈ {low, high}
- Letter       ∈ {weak, strong}

και τις ακμές:

- Difficulty → Grade
- Intelligence → Grade
- Intelligence → SAT
- Grade → Letter

Ο σκοπός του παραδείγματος είναι αποκλειστικά διδακτικός.
"""

from pathlib import Path

import joblib
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel


# Φάκελος για αποθήκευση του μοντέλου
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def build_student_bn():
    """
    Δημιουργεί και επιστρέφει ένα BayesianModel που αναπαριστά
    το Student Bayesian Network.

    Σημείωση για τη φιλοσοφία:

    - Στα Bayesian Networks, η κοινή κατανομή όλων των μεταβλητών
      παραγοντοποιείται ως γινόμενο τοπικών conditional πιθανοτήτων.

      Αν X = (X1, ..., Xn) και Pa(Xi) είναι οι γονείς του Xi, τότε:

          P(X1, ..., Xn) = Π_i P(Xi | Pa(Xi))

    - Το συγκεκριμένο δίκτυο είναι μικρό ώστε να μπορούμε να
      το αναλύσουμε "στο χέρι" και να δούμε πώς αλλάζουν
      οι posterior πιθανότητες όταν αλλάζει το evidence.
    """
    # Ορισμός της δομής (DAG) του δικτύου
    model = BayesianModel(
        [
            ("Difficulty", "Grade"),
            ("Intelligence", "Grade"),
            ("Intelligence", "SAT"),
            ("Grade", "Letter"),
        ]
    )

    # ---------------------------------------------------------
    # Ορισμός των CPDs (Conditional Probability Distributions)
    # ---------------------------------------------------------
    # Για κάθε κόμβο, ορίζουμε έναν πίνακα πιθανοτήτων TabularCPD.
    # Οι πιθανότητες είναι "λογικές" αλλά όχι από πραγματικά δεδομένα.
    # Χρησιμοποιούνται καθαρά για διδασκαλία.

    # Difficulty: prior P(Difficulty)
    cpd_difficulty = TabularCPD(
        variable="Difficulty",
        variable_card=2,
        values=[[0.6], [0.4]],
        state_names={"Difficulty": ["easy", "hard"]},
    )

    # Intelligence: prior P(Intelligence)
    cpd_intelligence = TabularCPD(
        variable="Intelligence",
        variable_card=2,
        values=[[0.7], [0.3]],
        state_names={"Intelligence": ["low", "high"]},
    )

    # Grade: P(Grade | Intelligence, Difficulty)
    # Σειρά evidence: [Intelligence, Difficulty]
    # Στήλες (I, D): (low, easy), (low, hard), (high, easy), (high, hard)
    cpd_grade = TabularCPD(
        variable="Grade",
        variable_card=3,
        values=[
            [0.30, 0.05, 0.90, 0.50],  # P(G=A | I,D)
            [0.40, 0.25, 0.08, 0.30],  # P(G=B | I,D)
            [0.30, 0.70, 0.02, 0.20],  # P(G=C | I,D)
        ],
        evidence=["Intelligence", "Difficulty"],
        evidence_card=[2, 2],
        state_names={
            "Grade": ["A", "B", "C"],
            "Intelligence": ["low", "high"],
            "Difficulty": ["easy", "hard"],
        },
    )

    # SAT: P(SAT | Intelligence)
    # Στήλες I=low, I=high
    cpd_sat = TabularCPD(
        variable="SAT",
        variable_card=2,
        values=[
            [0.95, 0.20],  # P(SAT=low  | I)
            [0.05, 0.80],  # P(SAT=high | I)
        ],
        evidence=["Intelligence"],
        evidence_card=[2],
        state_names={
            "SAT": ["low", "high"],
            "Intelligence": ["low", "high"],
        },
    )

    # Letter: P(Letter | Grade)
    # Στήλες G=A, G=B, G=C
    cpd_letter = TabularCPD(
        variable="Letter",
        variable_card=2,
        values=[
            [0.10, 0.40, 0.90],  # P(L=weak   | G)
            [0.90, 0.60, 0.10],  # P(L=strong | G)
        ],
        evidence=["Grade"],
        evidence_card=[3],
        state_names={
            "Letter": ["weak", "strong"],
            "Grade": ["A", "B", "C"],
        },
    )

    # Προσθέτουμε όλα τα CPDs στο μοντέλο
    model.add_cpds(
        cpd_difficulty,
        cpd_intelligence,
        cpd_grade,
        cpd_sat,
        cpd_letter,
    )

    # Έλεγχος ότι το μοντέλο είναι συνεπές
    model.check_model()

    return model


def save_model(model, filename: str = "student_bn.joblib"):
    """
    Αποθηκεύει το δοσμένο BayesianModel σε αρχείο .joblib
    μέσα στον φάκελο models/.
    """
    model_path = MODELS_DIR / filename
    joblib.dump(model, model_path)
    print(f"✅ Αποθηκεύτηκε το Bayesian Network στο: {model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Δημιουργία και αποθήκευση του Student Bayesian Network."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="student_bn.joblib",
        help="Όνομα .joblib αρχείου στο φάκελο models/.",
    )

    args = parser.parse_args()

    model = build_student_bn()
    print("Κόμβοι:", list(model.nodes()))
    print("Τόξα:", list(model.edges()))
    save_model(model, filename=args.output)
