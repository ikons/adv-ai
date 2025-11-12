# Φάκελος `data/`

Ο φάκελος `data/` περιέχει **μόνο** τα αρχεία δεδομένων (CSV κ.λπ.) που
κατεβάζουν τοπικά οι φοιτητές (π.χ. από Kaggle ή UCI). Τα μεγάλα αρχεία
**δεν** ανεβαίνουν στο GitHub.

---

## 0. Ρύθμιση Kaggle API (`kaggle.json`)

Για να χρησιμοποιήσετε τις εντολές `kaggle competitions download` και
`kaggle datasets download`, χρειάζεται ένα API token `kaggle.json`.

### 0.1 Δημιουργία `kaggle.json`

1. Δημιουργήστε λογαριασμό στο https://www.kaggle.com (αν δεν έχετε ήδη).
2. Κάντε login και πηγαίνετε στο προφίλ σας (**Profile**).
3. Επιλέξτε **Account** ή **My Account**.
4. Βρείτε την ενότητα **API**.
5. Πατήστε **Create New API Token**.
6. Θα κατέβει ένα αρχείο `kaggle.json` (συνήθως στον φάκελο `Downloads`).

### 0.2 Τοποθέτηση `kaggle.json` σε WSL / Linux / macOS

Αν δουλεύετε σε WSL, Ubuntu ή άλλο Linux/macOS:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

> Αν το αρχείο δεν βρίσκεται στο `~/Downloads`, προσαρμόστε το path ανάλογα.

### 0.3 Τοποθέτηση `kaggle.json` σε Windows (PowerShell / CMD)

1. Δημιουργήστε (αν δεν υπάρχει) τον φάκελο:

```text
C:\Users\<το-username-σας>\.kaggle\
```

2. Αντιγράψτε εκεί το `kaggle.json`:

```text
C:\Users\<το-username-σας>\.kaggle\kaggle.json
```

3. Βεβαιωθείτε ότι ο φάκελος και το αρχείο ανήκουν στον δικό σας χρήστη.

### 0.4 Έλεγχος εγκατάστασης

Μέσα στο virtual environment του repository:

```bash
kaggle competitions list
```

Αν όλα είναι σωστά ρυθμισμένα, η τελευταία εντολή θα εμφανίσει λίστα
με Kaggle competitions χωρίς error (το πακέτο kaggle έχει δηλωθεί στο requirements.txt επομένως έχει ήδη εγκατασταθεί).

---

## 1. Σύνολα δεδομένων που πρέπει να κατέβουν στον φάκελο `data/`

Τα παρακάτω datasets **δεν** φορτώνονται αυτόματα από scikit-learn/OpenML
και πρέπει να υπάρχουν ως αρχεία μέσα στο `data/`.

---

### 1.1 Titanic dataset (Kaggle)

Χρησιμοποιείται στα:

- `decision_trees/train_decision_tree_titanic.py`
- `notebooks/01_decision_trees_titanic.ipynb`

Αναμενόμενο όνομα αρχείου:

- `data/titanic_train.csv`

#### Τι περιέχουν τα δεδομένα;

Κάθε γραμμή αντιστοιχεί σε έναν επιβάτη του Titanic και περιλαμβάνει, μεταξύ άλλων:

- **Demographics & συνθήκες ταξιδιού**
  - `Pclass`: Κλάση εισιτηρίου (1η, 2η, 3η)
  - `Sex`: Φύλο
  - `Age`: Ηλικία
  - `SibSp`: Αριθμός αδερφών/συζύγων στο πλοίο
  - `Parch`: Αριθμός γονέων/παιδιών στο πλοίο
  - `Fare`: Τιμή εισιτηρίου
  - `Embarked`: Λιμάνι επιβίβασης (C, Q, S)

- **Στόχος (label)**
  - `Survived`: 0 = δεν επέζησε, 1 = επέζησε

#### Τι θα εξερευνήσουμε σε αυτό το dataset;

- Ποιοι παράγοντες σχετίζονται πιο έντονα με την **επιβίωση**:
  - Φύλο (π.χ. γυναίκες vs άνδρες)
  - Κλάση εισιτηρίου (1η vs 3η)
  - Τιμή εισιτηρίου (πιο ακριβό εισιτήριο → μεγαλύτερη πιθανότητα επιβίωσης;)
  - Ηλικία (παιδιά vs ενήλικες)
- Πώς ένα **δέντρο αποφάσεων** χωρίζει τους επιβάτες σε «επέζησε / δεν επέζησε».
- Πώς αλλάζει η απόδοση του μοντέλου όταν αλλάζουμε:
  - το **κριτήριο impurity** (`gini` vs `entropy`)
  - το **μέγιστο βάθος** (`max_depth`) → underfitting / overfitting.
- Ποιες στήλες το μοντέλο θεωρεί πιο «σημαντικές» (feature importances).

#### Λήψη με Kaggle CLI (WSL / Linux / macOS)

```bash
kaggle competitions download -c titanic -p data
unzip data/titanic.zip -d data

mv data/train.csv data/titanic_train.csv
```

#### Λήψη με Kaggle CLI (Windows / PowerShell)

```powershell
kaggle competitions download -c titanic -p data

Expand-Archive -Path data\titanic.zip -DestinationPath data
Rename-Item -Path data\train.csv -NewName titanic_train.csv
```

#### Χειροκίνητη λήψη

1. Μεταβείτε στη σελίδα του **Titanic** competition στο Kaggle.
2. Από το tab **Data**, κατεβάστε το `train.csv`.
3. Αποθηκεύστε το ως:

```text
data/titanic_train.csv
```

---

### 1.2 Mushroom Classification dataset (Kaggle ή UCI)

Χρησιμοποιείται στα:

- `decision_trees/impurity_measures_mushrooms.py`
- `notebooks/01b_impurity_measures_mushrooms.ipynb`

Αναμενόμενο όνομα αρχείου:

- `data/mushrooms.csv`

#### Τι περιέχουν τα δεδομένα;

Κάθε γραμμή είναι ένα μανιτάρι, με αποκλειστικά **κατηγορικά χαρακτηριστικά**, όπως:

- `cap-shape`, `cap-surface`, `cap-color`
- `odor` (οσμή)
- `gill-color`, `gill-size`, `gill-attachment`
- `stalk-shape`, `stalk-color` κ.ά.

και μια **κλάση-στόχο**:

- `class`:  
  - `e` = edible (βρώσιμο)  
  - `p` = poisonous (δηλητηριώδες)

#### Τι θα εξερευνήσουμε σε αυτό το dataset;

- Ποια χαρακτηριστικά (π.χ. **οσμή**, χρώμα καπέλου) είναι πιο «πληροφοριακά»  
  για να ξεχωρίσουμε **δηλητηριώδη από βρώσιμα** μανιτάρια.
- Πώς υπολογίζονται και πώς διαφέρουν μεταξύ τους:
  - **Information Gain (Entropy)**
  - **Split Information & Gain Ratio**
  - **Gini Gain**
- Πώς αυτά τα μέτρα μπορούν να οδηγήσουν ένα decision tree να επιλέξει
  **ποιο χαρακτηριστικό θα μπει πρώτο στη ρίζα** και ποια θα ακολουθήσουν.
- Πώς η επιλογή μέτρου impurity μπορεί να προκαλέσει bias
  (π.χ. υπέρ χαρακτηριστικών με πολλές κατηγορίες).

#### Λήψη με Kaggle CLI (WSL / Linux / macOS)

```bash
kaggle datasets download -d uciml/mushroom-classification -p data
unzip data/mushroom-classification.zip -d data

# Αν το αρχείο έχει άλλο όνομα (π.χ. mushroom.csv), μετονομάστε το:
# mv data/mushroom.csv data/mushrooms.csv
```

(Βεβαιωθείτε ότι τελικά το αρχείο ονομάζεται `mushrooms.csv`.)

#### Λήψη με Kaggle CLI (Windows / PowerShell)

```powershell
kaggle datasets download -d uciml/mushroom-classification -p data
Expand-Archive -Path data\mushroom-classification.zip -DestinationPath data

# Αν το αρχείο έχει όνομα mushroom.csv:
# Rename-Item -Path data\mushroom.csv -NewName mushrooms.csv
```

#### Χειροκίνητη λήψη (π.χ. από UCI)

1. Κατεβάστε το Mushroom dataset ως CSV (από Kaggle ή UCI mirror).
2. Αποθηκεύστε το ως:

```text
data/mushrooms.csv
```

---


### 1.3 House Prices (Kaggle)

Χρησιμοποιείται από:

- `regression/train_regression_house_prices.py`
- `notebooks/02_regression_house_prices.ipynb`
- `notebooks/02b_regularization_ridge_lasso.ipynb`

Αναμενόμενο αρχείο:

- `data/house_prices_train.csv`

#### Σύντομη περιγραφή του `house_prices_train.csv`

Το αρχείο περιλαμβάνει εγγραφές κατοικιών με την τιμή πώλησης (`SalePrice`) και πλήθος χαρακτηριστικών που περιγράφουν τα σπίτια.
Στα notebooks που συνοδεύουν αυτό το repo χρησιμοποιούμε ένα μικρό, εκπαιδευτικό υποσύνολο χαρακτηριστικών ώστε να δείξουμε βασικές τεχνικές regression και regularization:

- `LotArea` – μέγεθος οικοπέδου (τετραγωνικά πόδια)
- `OverallQual` – συνολική ποιότητα (βαθμός 1-10)
- `OverallCond` – συνολική κατάσταση (βαθμός 1-10)
- `YearBuilt` – έτος κατασκευής
- `GrLivArea` – κατοικήσιμη επιφάνεια (τετραγωνικά πόδια)
- `BedroomAbvGr` – αριθμός υπνοδωματίων πάνω από το έδαφος
- `GarageCars` – χωρητικότητα γκαράζ (αριθμός αυτοκινήτων)
- `SalePrice` – τιμή πώλησης (στόχος)

Τι εξετάζουμε:

- Πόσο καλά προβλέπεται η `SalePrice` από αυτό το σετ χαρακτηριστικών (MAE, RMSE, R²).
- Ποια χαρακτηριστικά έχουν μεγαλύτερο βάρος σε ένα γραμμικό μοντέλο.
- Πώς επηρεάζει η regularization (Ridge / Lasso) τους συντελεστές (shrinkage / sparsity) και την απόδοση στο test set.

Σημείωση: Το πλήρες dataset περιέχει πολλά επιπλέον χαρακτηριστικά· εδώ επιλέγουμε λίγα προς διδακτικούς σκοπούς. Για παραγωγική χρήση προτείνεται πιο εκτεταμένο feature engineering και cross-validation.

> **ΠΡΟΫΠΟΘΕΣΗ (για competitions):** Πριν δουλέψει το `kaggle` CLI, πρέπει να
> κάνεις **Join Competition / Accept Rules** στη σελίδα του διαγωνισμού (tab **Data**):
> https://www.kaggle.com/c/house-prices-advanced-regression-techniques  
> Αλλιώς θα πάρεις **403 Forbidden**.

Βήματα λήψης (αφού ρυθμίσετε το `kaggle.json` όπως στην ενότητα 0):

```bash
# 1. Κατεβάζουμε τα αρχεία του διαγωνισμού (zip)
kaggle competitions download -c house-prices-advanced-regression-techniques -p data

# 2. Αποσυμπίεση του zip
unzip data/house-prices-advanced-regression-techniques.zip -d data

# 3. Μετονομασία του train.csv σε πιο κατανοητό όνομα
mv data/train.csv data/house_prices_train.csv


### 1.4 Gas Sensor Array under Flow Modulation (UCI)

Χρησιμοποιείται από:

- `regression/gas_ridge_lasso_demo.py`
- `notebooks/03_regularization_gas_sensors.ipynb`

Αναμενόμενο αρχείο:

- `data/gas_features.csv`

#### Τι περιέχουν τα δεδομένα;

Πρόκειται για μετρήσεις από **16 αισθητήρες αερίων** υπό διαφορετικές ροές, για
μείγματα ακετόνης και αιθανόλης. Το αρχείο `features.csv` (από το UCI περιέχει):

- μεταδεδομένα πειράματος (`exp`, `batch`, `lab`, `gas`),
- συγκεντρώσεις στόχων:
  - `ace_conc` – συγκέντρωση ακετόνης (στόχος που χρησιμοποιούμε στο μάθημα)
  - `eth_conc` – συγκέντρωση αιθανόλης
- χαρακτηριστικά (features) από την απόκριση των αισθητήρων:
  - μέγιστες/ελάχιστες τιμές ανά αισθητήρα (`S1_max`, `S2_max`, …),
  - χαρακτηριστικά χαμηλής/υψηλής συχνότητας (π.χ. `S1_low`, `S1_high`, κ.λπ.).

Στο μάθημα χρησιμοποιούμε **όλα τα αριθμητικά χαρακτηριστικά** (p ≫ n)
για να δείξουμε πώς η Ridge και η Lasso βοηθούν όταν έχουμε
πολλά χαρακτηριστικά και λίγα δείγματα.

#### Λήψη από τη σελίδα UCI

1. Μεταβείτε στη σελίδα του dataset στο UCI:
   - "Gas sensor array under flow modulation".
2. Κατεβάστε το zip αρχείο `gas+sensor+array+under+flow+modulation.zip`.
3. Αποσυμπιέστε το και αντιγράψτε το `features.csv` στον φάκελο `data/` του repo.
4. (Προαιρετικά) μετονομάστε το σε πιο «μιλητό» όνομα:

```text
data/gas_features.csv
```

#### Λήψη με PowerShell (Windows)

Εναλλακτικά, μπορείτε να κατεβάσετε το zip απευθείας με PowerShell:

```powershell
# 1. Λήψη του zip από το UCI
Invoke-WebRequest `
  -Uri "https://archive.ics.uci.edu/static/public/308/gas+sensor+array+under+flow+modulation.zip" `
  -OutFile "data\gas_sensor_array_under_flow_modulation.zip"

# 2. Αποσυμπίεση του zip μέσα στο data/
Expand-Archive -Path "data\gas_sensor_array_under_flow_modulation.zip" -DestinationPath "data"

# 3. (Προαιρετικό) Μετονομασία του features.csv σε gas_features.csv
Rename-Item -Path "data\features.csv" -NewName "gas_features.csv"
```

Βεβαιωθείτε ότι τελικά υπάρχει ένα αρχείο:

```text
data/gas_features.csv
```

---





### 1.3 Μελλοντικά datasets (placeholders)

Στο μέλλον, ενδέχεται να προστεθούν επιπλέον datasets που θα αποθηκεύονται επίσης
στον φάκελο `data/`, π.χ.:

- Για `association_rules/` – market basket / retail data (π.χ. Online Retail).
- Για πιο “πλούσια” παραδείγματα regression (π.χ. House Prices).

Όταν υλοποιηθούν τα αντίστοιχα modules, εδώ θα προστεθούν συγκεκριμένες οδηγίες.

---

## 2. Σύνολα δεδομένων που φορτώνονται αυτόματα (χωρίς αρχεία στο `data/`)

Μερικά notebooks θα χρησιμοποιούν datasets τα οποία φορτώνονται απευθείας από
`scikit-learn` ή `OpenML`. Για αυτά **δεν χρειάζεται** χειροκίνητη λήψη ή αρχεία
στον φάκελο `data/` – η βιβλιοθήκη τα κατεβάζει και τα κάνει cache αυτόματα.

Ενδεικτικά παραδείγματα (προγραμματισμένα):

- `sklearn.datasets.load_iris()` – για k-NN, SVM, clustering.
- `sklearn.datasets.load_wine()` / `load_breast_cancer()` – για ταξινόμηση.
- `sklearn.datasets.make_blobs()` / `make_moons()` – για clustering & decision boundaries.
- `sklearn.datasets.fetch_california_housing()` – για regression παραδείγματα.
- `sklearn.datasets.fetch_openml("adult", version=2, as_frame=True)` – Adult income,
  για SVM / logistic regression / δέντρα αποφάσεων.

Όταν ένα notebook χρησιμοποιεί τέτοιο dataset, θα το αναφέρει ξεκάθαρα στην αρχή
(π.χ. “εδώ δεν χρειάζεται download, τα δεδομένα έρχονται από `sklearn.datasets`”).
