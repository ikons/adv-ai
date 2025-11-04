from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent / "models"


def load_model(model_name=None):
    """Load a trained decision tree model.

    By default it loads the 'gini, no depth limit' model.
    """
    if model_name is None:
        model_name = "decision_tree_titanic_gini_depthNone.joblib"
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file {model_path} not found. "
            "Train the model first with train_decision_tree_titanic.py."
        )
    return joblib.load(model_path)


def predict_example():
    pipe = load_model()

    # Example passenger (students can modify these values)
    example = {
        "Pclass": 3,
        "Sex": "female",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
    }

    X_new = pd.DataFrame([example])
    proba = pipe.predict_proba(X_new)[0]
    pred = pipe.predict(X_new)[0]

    print("Input features:", example)
    print(f"Predicted class: {int(pred)}  (0 = died, 1 = survived)")
    print(f"Class probabilities: {proba}")


if __name__ == "__main__":
    predict_example()
