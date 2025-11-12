"""
RIDGE vs LASSO - Gas Sensor Array under Flow Modulation (UCI)
==============================================================

Τι κάνει αυτό το script:
1. Φορτώνει τα features από το UCI dataset "Gas sensor array under flow modulation"
2. Ορίζει ως στόχο τη συγκέντρωση ακετόνης (ace_conc)
3. Εκπαιδεύει 3 μοντέλα: Linear, Ridge, Lasso
4. Συγκρίνει τα αποτελέσματα (MSE, RMSE, R²) σε train και test set

Γιατί είναι ενδιαφέρον:
- Έχουμε ΛΙΓΑ δείγματα και ΠΟΛΛΑ χαρακτηριστικά (p >> n),
  άρα η απλή OLS (Linear Regression) μπορεί να overfitάρει ή να είναι ασταθής.
- Η Ridge (L2) και η Lasso (L1) βοηθούν:
  * Ridge: "μαζεύει" τους συντελεστές, μειώνει το overfitting.
  * Lasso: μπορεί να μηδενίσει συντελεστές (feature selection).

Προϋποθέσεις:
- Αρχείο features από UCI μέσα στον φάκελο data/, π.χ.:
  data/gas_features.csv   (μετονομάζετε το features.csv από το UCI)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score


def main() -> None:
    print("=" * 70)
    print("RIDGE vs LASSO - Gas Sensor Array under Flow Modulation (UCI)")
    print("=" * 70)

    # ------------------------------------------------------------
    # ΒΗΜΑ 1: Φόρτωση features από data/
    # ------------------------------------------------------------
    print("\n📊 ΒΗΜΑ 1: Φόρτωση features από τον φάκελο data/...")

    data_path = Path("data") / "gas_features.csv"
    if not data_path.exists():
        # Fallback: αν ο φοιτητής δεν το μετονόμασε
        alt_path = Path("data") / "features.csv"
        if alt_path.exists():
            data_path = alt_path
        else:
            print("✗ Δεν βρέθηκε το αρχείο gas_features.csv ούτε features.csv στο data/.")
            print("  ➜ Κατεβάστε το zip από το UCI, αποσυμπιέστε το μέσα στο data/")
            print("    και μετονομάστε το features.csv σε gas_features.csv.")
            return

    print(f"✓ Φορτώνουμε από: {data_path}")
    df = pd.read_csv(data_path)

    print(f"  • Δείγματα: {df.shape[0]}")
    print(f"  • Στήλες  : {df.shape[1]}")
    print("  • Πρώτες στήλες:", ", ".join(df.columns[:8]), "...")

    # ------------------------------------------------------------
    # ΒΗΜΑ 2: Ορισμός στόχου (target) και χαρακτηριστικών
    # ------------------------------------------------------------
    print("\n🎯 ΒΗΜΑ 2: Ορισμός στόχου και χαρακτηριστικών...")

    target_col = "ace_conc"  # συγκέντρωση ακετόνης
    if target_col not in df.columns:
        print(f"✗ Δεν βρέθηκε η στήλη-στόχος '{target_col}' στο DataFrame.")
        print("  Διαθέσιμες στήλες:")
        for col in df.columns:
            print("   -", col)
        return

    y = df[target_col].astype("float64").values

    # Κρατάμε μόνο αριθμητικά features, εκτός από τον στόχο
    X_num = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    print(f"  • Αριθμητικά χαρακτηριστικά που θα χρησιμοποιηθούν: {X_num.shape[1]}")
    print("    Πρώτες numeric στήλες:", ", ".join(X_num.columns[:8]), "...")

    X = X_num.astype("float64").values

    print(f"  • Στόχος (y): {target_col}")
    print(f"  • Χαρακτηριστικά (X): {X.shape[1]} columns")
    print(f"  • Δείγματα          : {X.shape[0]} rows")

    # ------------------------------------------------------------
    # ΒΗΜΑ 3: Train/Test split
    # ------------------------------------------------------------
    print("\n🔧 ΒΗΜΑ 3: Διαχωρισμός σε train/test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"  • Train: {len(X_train)} δείγματα")
    print(f"  • Test : {len(X_test)} δείγματα")

    # ------------------------------------------------------------
    # ΒΗΜΑ 4: Ορισμός μοντέλων (Scaling → Model)
    # ------------------------------------------------------------
    print("\n🤖 ΒΗΜΑ 4: Ορισμός μοντέλων...")

    alphas_ridge = np.logspace(-3, 3, 13)
    alphas_lasso = np.logspace(-3, 3, 13)

    models = {
        "Linear (χωρίς κανονικοποίηση)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge (L2 κανονικοποίηση)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=alphas_ridge, cv=5)),
            ]
        ),
        "Lasso (L1 κανονικοποίηση)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LassoCV(alphas=alphas_lasso, cv=5, max_iter=10000)),
            ]
        ),
    }

    # ------------------------------------------------------------
    # ΒΗΜΑ 5: Εκπαίδευση & Αξιολόγηση
    # ------------------------------------------------------------
    print("\n🏋️ ΒΗΜΑ 5: Εκπαίδευση μοντέλων και αξιολόγηση στο test...")

    results = []

    for name, pipe in models.items():
        print(f"\n  → {name}")
        pipe.fit(X_train, y_train)

        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)

        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        gap = rmse_test - rmse_train

        print(f"    MSE  (Train): {mse_train:.4f} | MSE  (Test): {mse_test:.4f}")
        print(f"    RMSE (Train): {rmse_train:.4f} | RMSE (Test): {rmse_test:.4f}")
        print(f"    R²   (Train): {r2_train:.4f} | R²  (Test): {r2_test:.4f}")
        print(f"    Gap (Test - Train): {gap:.4f}  (μικρότερο = λιγότερο overfitting)")

        alpha = None
        model = pipe.named_steps["model"]
        if hasattr(model, "alpha_"):
            alpha = float(model.alpha_)
            print(f"    Επιλεγμένο alpha: {alpha:.6f}")

        results.append(
            {
                "Μοντέλο": name.split("(")[0].strip(),
                "MSE_Train": mse_train,
                "MSE_Test": mse_test,
                "RMSE_Train": rmse_train,
                "RMSE_Test": rmse_test,
                "Gap_Test_Train": gap,
                "R2_Test": r2_test,
                "Alpha": alpha,
            }
        )

    # ------------------------------------------------------------
    # ΒΗΜΑ 6: Σύγκριση μοντέλων
    # ------------------------------------------------------------
    print("\n📈 ΒΗΜΑ 6: Σύγκριση μοντέλων")
    print("=" * 70)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("=" * 70)

    best_rmse = results_df.loc[results_df["RMSE_Test"].idxmin()]
    best_gap = results_df.loc[results_df["Gap_Test_Train"].abs().idxmin()]

    print(
        "\n✓ Καλύτερο Test RMSE: "
        f"{best_rmse['Μοντέλο']} "
        f"(RMSE_Test={best_rmse['RMSE_Test']:.4f}, R²_Test={best_rmse['R2_Test']:.4f})"
    )
    print(
        "✓ Λιγότερο overfitting (μικρότερο |Gap|): "
        f"{best_gap['Μοντέλο']} "
        f"(|Gap|={abs(best_gap['Gap_Test_Train']):.4f})"
    )

    print("\n💡 Υπενθύμιση για το μάθημα:")
    print("  - Εδώ έχουμε λίγα δείγματα και πολλά χαρακτηριστικά (p >> n).")
    print("  - Η OLS (Linear) μπορεί να εμφανίζει σχεδόν μηδενικό σφάλμα στο train,")
    print("    αλλά χειρότερη γενίκευση στο test.")
    print("  - Η Ridge (L2) 'μαζεύει' τους συντελεστές και σταθεροποιεί το μοντέλο.")
    print("  - Η Lasso (L1) μπορεί να μηδενίσει βάρη και να κάνει feature selection.")


if __name__ == "__main__":
    main()
