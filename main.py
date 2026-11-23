from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer.csv"


def main():
    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Unnamed: 32"], errors="ignore")

    if "diagnosis" not in df.columns:
        raise ValueError("The CSV must contain a 'diagnosis' target column.")

    # 2. Prepare features and target
    # Remove ID because it is an identifier, not a useful measurement.
    X = df.drop(columns=["id", "diagnosis"], errors="ignore")
    y = df["diagnosis"]

    # Ensure feature columns are numeric and handle missing feature values.
    X = X.apply(pd.to_numeric, errors="coerce")
    valid_rows = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if X.empty:
        raise ValueError("No usable rows remain after cleaning the dataset.")

    print("Dataset shape:", df.shape)
    print("Feature matrix shape:", X.shape)
    print("\nClass counts:")
    print(y.value_counts())

    # 3. Split before scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    # 4. Standardize features: fit only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train KNN
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    # 6. Predict test set
    y_pred = knn.predict(X_test_scaled)

    # 7. Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nK value: {k}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (labels order:", labels, "):")
    print(cm)

    # Save/display confusion matrix
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"KNN Confusion Matrix (K={k})")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.show()

    # 8. Compare multiple K values on the same held-out test set
    print("\nAccuracy for different K values:")
    k_values = [3, 5, 7, 9, 11]
    results = []
    for current_k in k_values:
        if current_k > len(X_train):
            continue
        model = KNeighborsClassifier(n_neighbors=current_k)
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        score = accuracy_score(y_test, predictions)
        results.append({"K": current_k, "Accuracy": score})
        print(f"K={current_k:>2}: {score:.4f}")

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "k_comparison.csv", index=False)

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(7, 4))
        sns.lineplot(data=results_df, x="K", y="Accuracy", marker="o")
        plt.title("KNN Accuracy for Different K Values")
        plt.ylabel("Test accuracy")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(output_dir / "k_comparison.png", dpi=150)
        plt.show()

    print(f"\nOutputs saved in: {output_dir}")
    print("This is an educational machine-learning project, not a medical diagnostic tool.")


if __name__ == "__main__":
    main()
