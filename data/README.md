# `data/` directory

The `data/` folder contains the CSV and other data files that students
download locally (e.g., from Kaggle). Large data files are **not**
committed to the repository.

This folder is ignored by git, except for this `README.md` and `.gitignore`.

## Git ignore rules

At the repository root in `.gitignore` we have:

```gitignore
data/*
!data/README.md
!data/.gitignore
```

And here in `data/.gitignore`:

```gitignore
*
!README.md
!.gitignore
```

This means students can place any number of files here, but they will not
be committed to GitHub.

---

## 1. Titanic dataset

Used in:

- `decision_trees/train_decision_tree_titanic.py`
- `notebooks/01_decision_trees_titanic.ipynb`

Expected file name:

- `data/titanic_train.csv`

### 1.1 With Kaggle CLI (WSL / Linux / macOS)

1. Create a Kaggle account and set up the `kaggle.json` API token under `~/.kaggle/`.
2. From the root of this repository:

```bash
pip install kaggle
cd path/to/advanced-ml-tutorials
mkdir -p data

kaggle competitions download -c titanic -p data
unzip data/titanic.zip -d data

mv data/train.csv data/titanic_train.csv
```

### 1.2 With Kaggle CLI (Windows / PowerShell)

```powershell
pip install kaggle
cd path\to\advanced-ml-tutorials
mkdir data

kaggle competitions download -c titanic -p data

Expand-Archive -Path data\titanic.zip -DestinationPath data
Rename-Item -Path data\train.csv -NewName titanic_train.csv
```

### 1.3 Manual download

1. Open the Titanic competition page on Kaggle.
2. Download `train.csv` from the **Data** tab.
3. Save it as:

```text
data/titanic_train.csv
```

---

## 2. Mushroom Classification dataset

Used in:

- `decision_trees/impurity_measures_mushrooms.py`
- (optional) `notebooks/01b_impurity_measures_mushrooms.ipynb`

Expected file name:

- `data/mushrooms.csv`

### 2.1 With Kaggle CLI (WSL / Linux / macOS)

```bash
pip install kaggle
cd path/to/advanced-ml-tutorials
mkdir -p data

kaggle datasets download -d uciml/mushroom-classification -p data
unzip data/mushroom-classification.zip -d data

# If the file name differs (e.g., mushroom.csv), rename it:
mv data/mushrooms.csv data/mushrooms.csv  # adjust if needed
```

### 2.2 With Kaggle CLI (Windows / PowerShell)

```powershell
pip install kaggle
cd path\to\advanced-ml-tutorials
mkdir data

kaggle datasets download -d uciml/mushroom-classification -p data
Expand-Archive -Path data\mushroom-classification.zip -DestinationPath data

# If the file is called mushroom.csv, rename it:
# Rename-Item -Path data\mushroom.csv -NewName mushrooms.csv
```

### 2.3 Manual download

1. Open the **Mushroom Classification** dataset on Kaggle.
2. Download the main CSV file.
3. Save it as:

```text
data/mushrooms.csv
```

---

## 3. Future datasets (placeholders)

For the following modules, additional datasets will be added later:

- `regression/` (e.g. House Prices)
- `bayesian_learning/` (e.g. spam classification or reuse Titanic)
- `bayesian_networks/` (small medical / toy examples)
- `association_rules/` (market basket / retail)
- `knn/`
- `svm/`
- `clustering/`

When these modules are implemented, this `README.md` will be updated
with specific Kaggle links and file names.
