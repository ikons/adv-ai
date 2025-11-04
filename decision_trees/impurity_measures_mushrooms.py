from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "mushrooms.csv"


def entropy(y: pd.Series) -> float:
    p = y.value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum())


def gini_impurity(y: pd.Series) -> float:
    p = y.value_counts(normalize=True)
    return float(1.0 - (p**2).sum())


def feature_scores(df: pd.DataFrame, feature: str, target: str):
    """Compute IG, SplitInfo, GainRatio and Gini gain for a categorical feature."""
    y = df[target]
    X = df[feature]

    n = len(df)

    H_before = entropy(y)
    G_before = gini_impurity(y)

    H_after = 0.0
    G_after = 0.0
    split_info = 0.0

    for v, subset in df.groupby(feature):
        weight = len(subset) / n
        y_sub = subset[target]

        H_after += weight * entropy(y_sub)
        G_after += weight * gini_impurity(y_sub)

        if weight > 0:
            split_info -= weight * np.log2(weight)

    info_gain = H_before - H_after
    gini_gain = G_before - G_after
    gain_ratio = info_gain / split_info if split_info > 0 else 0.0

    return info_gain, split_info, gain_ratio, gini_gain


def main():
    df = pd.read_csv(DATA_PATH)

    target = "class"  # 'e' or 'p'
    features = [c for c in df.columns if c != target]

    rows = []
    for feat in features:
        ig, si, gr, gg = feature_scores(df, feat, target)
        rows.append(
            {
                "feature": feat,
                "num_values": df[feat].nunique(),
                "info_gain": ig,
                "split_info": si,
                "gain_ratio": gr,
                "gini_gain": gg,
            }
        )

    scores = pd.DataFrame(rows)

    print("\n=== Top features by Information Gain ===")
    print(scores.sort_values("info_gain", ascending=False).head(10))

    print("\n=== Top features by Gain Ratio ===")
    print(scores.sort_values("gain_ratio", ascending=False).head(10))

    print("\n=== Top features by Gini Gain ===")
    print(scores.sort_values("gini_gain", ascending=False).head(10))


if __name__ == "__main__":
    main()
