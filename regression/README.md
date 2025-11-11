# Παλινδρόμηση (Regression)

Σε αυτό το module υλοποιούμε **μοντέλα παλινδρόμησης** για πρόβλεψη συνεχών
μεταβλητών (π.χ. τιμή κατοικίας) με τη βιβλιοθήκη `scikit-learn`.

## Στόχοι

- Κατανόηση της βασικής **γραμμικής παλινδρόμησης**.
- Εισαγωγή στην **κανονικοποίηση (regularization)**:
  - Ridge Regression (L2)
  - Lasso Regression (L1)
- Χρήση `Pipeline` με:
  - `StandardScaler` για κλιμάκωση χαρακτηριστικών
  - γραμμικά μοντέλα παλινδρόμησης (`LinearRegression`, `Ridge`, `Lasso`)
- Ερμηνεία βασικών μετρικών:
  - MAE (Mean Absolute Error)
  - MSE / RMSE (Mean Squared Error / Root MSE)
  - R² (συντελεστής προσδιορισμού)

---

## Αρχεία Python (`regression/`)

### 1. `train_regression_house_prices.py`

Script για εκπαίδευση 3 μοντέλων στο dataset **House Prices** (Kaggle):

- Γραμμική Παλινδρόμηση (`LinearRegression`)
- Ridge (`Ridge`)
- Lasso (`Lasso`)

Τι κάνει:

1. Φορτώνει το αρχείο `data/house_prices_train.csv`.
2. Επιλέγει ένα υποσύνολο αριθμητικών χαρακτηριστικών (LotArea, OverallQual, κ.λπ.).
3. Χειρίζεται τις ελλιπείς τιμές με διάμεσο (median).
4. Διαχωρίζει σε train / test.
5. Εκπαιδεύει 3 pipelines:
   - `StandardScaler` → μοντέλο παλινδρόμησης.
6. Υπολογίζει MAE, RMSE, R² στο test set.
7. Επιλέγει το καλύτερο μοντέλο (μικρότερο RMSE) και το αποθηκεύει σε:
   - `regression/models/regression_house_prices_<model>.joblib`
8. Αποθηκεύει διάγραμμα **πραγματικών vs προβλεπόμενων** τιμών:
   - `regression/models/regression_house_prices_<model>_scatter.png`

Χρήση από γραμμή εντολών (παράδειγμα):

```bash
python -m regression.train_regression_house_prices --test_size 0.2 --alpha_ridge 1.0 --alpha_lasso 0.1
```

---

### 2. `infer_regression_house_prices.py`

Μικρό script για **inference** (πρόβλεψη) με ένα εκπαιδευμένο μοντέλο.

Τι κάνει:

1. Φορτώνει το καλύτερο διαθέσιμο `.joblib` από τον φάκελο `regression/models/`.
2. Δημιουργεί ένα παράδειγμα σπιτιού (dictionary με LotArea, OverallQual, κ.λπ.).
3. Μετατρέπει το παράδειγμα σε `DataFrame`.
4. Εκτυπώνει την προβλεπόμενη τιμή `SalePrice`.

Χρήση:

```bash
python -m regression.infer_regression_house_prices
```

Μπορείτε να αλλάξετε τις τιμές στο `example` dictionary και να δείτε
πώς αλλάζει η προβλεπόμενη τιμή.
