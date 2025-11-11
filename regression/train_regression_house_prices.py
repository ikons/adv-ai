from pathlib import Path
import argparse

import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Ρίζα repository (ένας φάκελος πάνω από τον τρέχοντα: regression/)
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'house_prices_train.csv'
MODELS_DIR = Path(__file__).resolve().parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)


def load_house_prices():
    # Φόρτωση CSV και απλή προεπεξεργασία.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f'Δεν βρέθηκε το αρχείο δεδομένων: {DATA_PATH}\n'
            'Κατέβασε το dataset House Prices από το Kaggle και αποθήκευσέ το ως data/house_prices_train.csv.'
        )

    df = pd.read_csv(DATA_PATH)

    feature_names = [
        'LotArea',
        'OverallQual',
        'OverallCond',
        'YearBuilt',
        'GrLivArea',
        'BedroomAbvGr',
        'GarageCars',
    ]
    target = 'SalePrice'

    df = df[feature_names + [target]].copy()

    # Απλή διαχείριση ελλιπών τιμών
    df = df.dropna(subset=[target])
    df[feature_names] = df[feature_names].fillna(df[feature_names].median())

    X = df[feature_names]
    y = df[target]
    return X, y, feature_names


def evaluate_regression_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}


def train_models(test_size=0.2, random_state=0, alpha_ridge=1.0, alpha_lasso=0.1):
    # Φόρτωση δεδομένων
    X, y, feature_names = load_house_prices()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression()),
        ]),
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=alpha_ridge, random_state=random_state)),
        ]),
        'Lasso': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=alpha_lasso, random_state=random_state)),
        ]),
    }

    results = []
    trained_models = {}

    for name, pipe in models.items():
        print(f"\n=== Εκπαίδευση μοντέλου: {name} ===")
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe
        y_pred_test = pipe.predict(X_test)
        stats = evaluate_regression_model(name, y_test, y_pred_test)
        results.append(stats)
        print(f"MAE  (|y-ŷ| μέσο):      {stats['MAE']:.2f}")
        print(f"RMSE (ρίζα MSE):       {stats['RMSE']:.2f}")
        print(f"R^2 (συντ. προσδιορ.): {stats['R2']:.4f}")

    results_df = pd.DataFrame(results).set_index('model')
    print('\n=== Συνοπτικά αποτελέσματα (test set) ===')
    print(results_df)

    best_model_name = results_df['RMSE'].idxmin()
    best_model = trained_models[best_model_name]
    print(f"\nΚαλύτερο μοντέλο (με βάση το RMSE): {best_model_name}")

    model_path = MODELS_DIR / f'regression_house_prices_{best_model_name.lower()}.joblib'
    joblib.dump(best_model, model_path)
    print(f'Αποθηκεύτηκε το μοντέλο στο αρχείο: {model_path}')

    # Δημιουργία scatter plots για όλα τα μοντέλα
    print(f"\n=== Δημιουργία scatter plots για όλα τα μοντέλα ===")
    for model_name, pipe in trained_models.items():
        y_pred = pipe.predict(X_test)
        
        plt.figure(figsize=(7, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, s=40)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        # Υπολογισμός μετρικών για το title
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        plt.xlabel('Πραγματική τιμή (SalePrice)', fontsize=11)
        plt.ylabel('Προβλεπόμενη τιμή (SalePrice)', fontsize=11)
        plt.title(f'House Prices – {model_name}\nR² = {r2:.4f}, RMSE = {rmse:.0f}, MAE = {mae:.0f}', 
                  fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        fig_path = MODELS_DIR / f'regression_house_prices_{model_name.lower()}_scatter.png'
        plt.savefig(fig_path, dpi=150)
        print(f'Αποθηκεύτηκε το διάγραμμα για {model_name} στο: {fig_path}')
        plt.close()
    
    return best_model_name, results_df


def main():
    parser = argparse.ArgumentParser(
        description='Εκπαίδευση Linear/Ridge/Lasso στο dataset House Prices (Kaggle).'
    )
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--alpha_ridge', type=float, default=1.0)
    parser.add_argument('--alpha_lasso', type=float, default=0.1)
    args = parser.parse_args()
    train_models(
        test_size=args.test_size,
        random_state=args.random_state,
        alpha_ridge=args.alpha_ridge,
        alpha_lasso=args.alpha_lasso,
    )


if __name__ == '__main__':
    main()
