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


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "titanic_train.csv"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_titanic():
    """Load Titanic training data and return (X, y, preprocessor, cat_cols, num_cols)."""
    df = pd.read_csv(DATA_PATH)

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    target = "Survived"

    df = df[features + [target]]

    # Simple imputations
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    X = df[features]
    y = df[target]

    cat_cols = ["Sex", "Embarked"]
    num_cols = [c for c in features if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    return X, y, preprocessor, cat_cols, num_cols


def train(criterion="gini", max_depth=None, random_state=0, test_size=0.2):
    X, y, preprocessor, cat_cols, num_cols = load_titanic()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    print(f"=== Decision Tree (criterion={criterion}, max_depth={max_depth}) ===")
    print(classification_report(y_val, y_pred, digits=3))

    # Save model
    model_file = (
        MODELS_DIR
        / f"decision_tree_titanic_{criterion}_depth{max_depth or 'None'}.joblib"
    )
    joblib.dump(pipe, model_file)
    print(f"Saved model to {model_file}")

    # Optional: plot tree when depth is small
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a decision tree on Titanic data.")
    parser.add_argument(
        "--criterion",
        choices=["gini", "entropy"],
        default="gini",
        help="Impurity measure to use.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Maximum tree depth (None = no limit).",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Validation set size as a fraction.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    train(
        criterion=args.criterion,
        max_depth=args.max_depth,
        random_state=args.random_state,
        test_size=args.test_size,
    )
