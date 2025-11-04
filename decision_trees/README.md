# Decision Trees (Classification)

This module demonstrates **decision tree classifiers** for supervised
learning (classification) using Python & scikit-learn.

Topics linked to the course:

- Decision trees (C4.5/CART-style)
- Purity measures:
  - Information Gain (entropy)
  - Gain Ratio (theoretical, illustrated via custom code)
  - Gini Index
- Biases of each measure
- Overfitting vs Underfitting via tree depth
- Handling categorical vs continuous features

## Files

- `train_decision_tree_titanic.py`  
  Train a decision tree classifier on the Titanic dataset.

- `infer_decision_tree_titanic.py`  
  Load a trained model and run inference on a single passenger.

- `impurity_measures_mushrooms.py`  
  Compute Information Gain, Split Information, Gain Ratio and Gini gain
  for each feature in the Mushroom dataset.

- `models/`  
  Stores trained models (`.joblib`) and optional tree plots (`.png`).

## Datasets

Expected files in `data/` (see `data/README.md`):

- `titanic_train.csv`  (Titanic competition training data)
- `mushrooms.csv`      (Mushroom Classification dataset)

## Running the scripts

From the root of the repository (after placing the datasets):

```bash
# Train decision tree on Titanic (Gini, max_depth=4)
python -m decision_trees.train_decision_tree_titanic --criterion gini --max_depth 4

# Train decision tree on Titanic (Entropy, max_depth=4)
python -m decision_trees.train_decision_tree_titanic --criterion entropy --max_depth 4

# Inference using the default saved model
python -m decision_trees.infer_decision_tree_titanic

# Compute impurity measures on Mushroom dataset
python -m decision_trees.impurity_measures_mushrooms
```

## Notebooks

- `notebooks/01_decision_trees_titanic.ipynb`  
  Step-by-step training, evaluation and inference on Titanic.

- `notebooks/01b_impurity_measures_mushrooms.ipynb`  
  (Placeholder) Will illustrate purity measures on the Mushroom dataset.
