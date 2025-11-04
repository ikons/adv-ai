# Advanced AI / Advanced Machine Learning Tutorials

This repository contains simple, well-documented Python implementations
of the algorithms taught in the **Advanced Artificial Intelligence**
course at the **Dept. of Informatics & Telecommunications, University of Thessaly**.

Goal: help students bridge **theory → practice** by providing:

- Jupyter notebooks with step-by-step training & inference.
- Python scripts (`.py`) that show cleaner, reusable code.

## Structure

- `decision_trees/` – Decision Trees for classification (Titanic, Mushrooms)
- `regression/` – Regression (planned)
- `bayesian_learning/` – Naive Bayes & Bayesian learning (planned)
- `bayesian_networks/` – Bayesian Belief Networks (planned)
- `association_rules/` – Association rules & Apriori (planned)
- `knn/` – k-Nearest Neighbors (planned)
- `svm/` – Support Vector Machines (planned)
- `clustering/` – Clustering (k-means, BIRCH, DBSCAN, silhouette) (planned)
- `notebooks/` – Jupyter notebooks, one per topic
- `data/` – Local data files (not tracked by git)

Large data files are **not** committed to git. Only `data/README.md` and
`data/.gitignore` are tracked.

## Installation

```bash
git clone https://github.com/ikons/adv-ai.git
cd adv-ai

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

## Running the notebooks

From the root of the repo:

```bash
jupyter notebook
# or
jupyter lab
```

Then open `notebooks/` and start with:

- `01_decision_trees_titanic.ipynb`

Each notebook explains which dataset it needs and how to run the experiments.

## Data

Datasets are downloaded from Kaggle (or loaded via `sklearn.datasets`).
See `data/README.md` for dataset-specific instructions.
