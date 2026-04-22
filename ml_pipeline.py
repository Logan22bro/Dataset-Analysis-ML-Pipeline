"""
Dataset Analysis ML Pipeline

This Pipeline:
- loads tabular data from data.txt
- explores the raw dataset
- visualizes major and label distributions
- cleans and encodes features
- scales numeric features
- trains/evaluates classifiers with cross-validation
- performs simple significance tests between model score sets
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_data(path: str = "data.txt") -> pd.DataFrame:
    """Load the tab-separated dataset."""
    return pd.read_csv(path, sep="\t")


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset and one-hot encode text columns."""
    clean_frame = frame.copy()

    clean_frame = clean_frame.replace(
        ['None', 'NONE', 'N/A', 'n/a', 'NULL', 'null', '?', '', ' '],
        np.nan
    )

    text_cols = []

    for col in clean_frame.columns:
        if col == "label":
            continue

        vals = clean_frame[col]

        numerics = pd.to_numeric(vals, errors="coerce")
        if numerics.notna().sum() == vals.notna().sum():
            clean_frame[col] = numerics
            continue

        extracted = vals.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
        extracted_nums = pd.to_numeric(extracted, errors="coerce")
        if extracted_nums.notna().sum() == vals.notna().sum():
            clean_frame[col] = extracted_nums
            continue

        clean_frame[col] = vals.astype(str).str.strip().str.lower()
        clean_frame[col] = clean_frame[col].replace("nan", np.nan)
        text_cols.append(col)

    if "label" in clean_frame.columns:
        clean_frame = clean_frame.dropna(subset=["label"])

    for col in text_cols:
        if col in clean_frame.columns:
            dummies = clean_frame[col].str.get_dummies(sep=",")
            dummies.columns = [f"{col}: {value.strip()}" for value in dummies.columns]
            clean_frame = pd.concat([clean_frame, dummies], axis=1)
            clean_frame = clean_frame.drop(columns=[col])

    clean_frame = clean_frame.fillna(0)
    return clean_frame


def scale_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Scale all feature columns while preserving the label column."""
    scaled_frame = frame.copy()

    X = scaled_frame.drop(columns=["label"])
    y = scaled_frame["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaled_frame = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    scaled_frame["label"] = y
    return scaled_frame


def create_classifiers():
    """Return the list of classifiers used in the notebook."""
    return [
        LogisticRegression(max_iter=1000),
        KNeighborsClassifier(n_neighbors=15),
        DecisionTreeClassifier()
    ]


def cross_fold_validation(classifier, frame: pd.DataFrame, folds: int):
    """Run k-fold cross-validation and return accuracy scores as floats."""
    X = frame.drop(columns=["label"])
    y = frame["label"]
    scores = cross_val_score(classifier, X, y, cv=folds, scoring="accuracy")
    return [float(score) for score in scores]


def significance_test(a_values, b_values, p_value: float) -> bool:
    """Return True if the difference is significant under the given threshold."""
    _, p = ttest_ind(a_values, b_values)
    return p < p_value


def plot_top_majors(frame: pd.DataFrame):
    """Plot the top 5 majors from col_02 if the column exists."""
    if "col_02" not in frame.columns:
        print("Skipping major plot: 'col_02' not found.")
        return

    major_count = frame["col_02"].value_counts().head(5)
    major_count.plot(kind="bar", title="Top 5 Academic Majors")
    plt.tight_layout()
    plt.show()


def plot_label_distribution(frame: pd.DataFrame):
    """Plot the label distribution if the label column exists."""
    if "label" not in frame.columns:
        print("Skipping label plot: 'label' not found.")
        return

    count = frame["label"].value_counts().sort_index()
    count.plot(kind="bar", title="Label Distribution")
    plt.tight_layout()
    plt.show()


def main():
    unique_data = load_data("data.txt")

    print("\n=== RAW INFO ===")
    unique_data.info()

    print("\n=== RAW DESCRIPTION ===")
    print(unique_data.describe(include="all"))

    plot_top_majors(unique_data)

    unique_data = clean_data(unique_data)

    print("\n=== CLEANED INFO ===")
    unique_data.info()

    plot_label_distribution(unique_data)

    scaled_data = scale_features(unique_data)

    classifiers = create_classifiers()

    print("\n=== CROSS-VALIDATION SCORES ===")
    classifier_scores = []
    for classifier in classifiers:
        accuracy_scores = cross_fold_validation(classifier, scaled_data, 5)
        classifier_scores.append(accuracy_scores)
        print(f"Classifier: {type(classifier).__name__}, Accuracy: {accuracy_scores}")

    print("\n=== SIGNIFICANCE TESTS (p < 0.10) ===")
    for i in range(len(classifiers)):
        for j in range(i + 1, len(classifiers)):
            significant = significance_test(classifier_scores[i], classifier_scores[j], 0.10)
            print(f"{type(classifiers[i]).__name__} vs {type(classifiers[j]).__name__}: {significant}")


if __name__ == "__main__":
    main()
