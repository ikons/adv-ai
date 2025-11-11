from pathlib import Path

import joblib
import pandas as pd


MODELS_DIR = Path(__file__).resolve().parent / 'models'


def load_model(model_filename=None):
    # Φόρτωση εκπαιδευμένου pipeline παλινδρόμησης από τον φάκελο models.
    if model_filename is None:
        candidates = [
            'regression_house_prices_linearregression.joblib',
            'regression_house_prices_ridge.joblib',
            'regression_house_prices_lasso.joblib',
        ]
        for fname in candidates:
            path = MODELS_DIR / fname
            if path.exists():
                model_filename = fname
                break
    if model_filename is None:
        raise FileNotFoundError(
            'Δεν βρέθηκε εκπαιδευμένο μοντέλο. '
            'Τρέξε πρώτα το train_regression_house_prices.py για να δημιουργήσεις ένα .joblib αρχείο.'
        )
    model_path = MODELS_DIR / model_filename
    print(f'Φόρτωση μοντέλου από: {model_path}')
    pipe = joblib.load(model_path)
    return pipe


def predict_example():
    # Παράδειγμα πρόβλεψης τιμής κατοικίας.
    pipe = load_model()

    example = {
        'LotArea': 8000,
        'OverallQual': 7,
        'OverallCond': 5,
        'YearBuilt': 2005,
        'GrLivArea': 1600,
        'BedroomAbvGr': 3,
        'GarageCars': 2,
    }

    X_new = pd.DataFrame([example])
    y_pred = pipe.predict(X_new)[0]

    print('Χαρακτηριστικά εισόδου:')
    for k, v in example.items():
        print(f'  {k}: {v}')

    print(f"\nΠροβλεπόμενη τιμή πώλησης (SalePrice): {y_pred:,.2f} $ (περίπου)")


if __name__ == '__main__':
    predict_example()
