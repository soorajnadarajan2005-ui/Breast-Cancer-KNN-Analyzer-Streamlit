from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer.csv"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "eda_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove common empty export column, if present
df = df.drop(columns=["Unnamed: 32"], errors="ignore")

print("=" * 60)
print("BREAST CANCER DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

if "diagnosis" not in df.columns:
    raise ValueError("The CSV must contain a 'diagnosis' column.")

print("\nDiagnosis counts:")
print(df["diagnosis"].value_counts())

print("\nDiagnosis proportions:")
print(df["diagnosis"].value_counts(normalize=True).round(3))

# Descriptive statistics for numeric columns
print("\nNumerical descriptive statistics:")
print(df.describe().T)

# Plot style
sns.set_theme(style="whitegrid")

# 1. Diagnosis count plot
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="diagnosis")
plt.title("Diagnosis Class Distribution")
plt.xlabel("Diagnosis (B = Benign, M = Malignant)")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "diagnosis_count.png", dpi=150)
plt.show()

# 2. Distribution of selected features
selected_features = [
    col for col in ["radius_mean", "texture_mean", "area_mean", "smoothness_mean"]
    if col in df.columns
]
if selected_features:
    df[selected_features].hist(figsize=(10, 7), bins=20, edgecolor="black")
    plt.suptitle("Distribution of Selected Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=150)
    plt.show()

# 3. Correlation heatmap for numeric features (excluding ID)
numeric_df = df.select_dtypes(include="number").drop(columns=["id"], errors="ignore")
if not numeric_df.empty:
    plt.figure(figsize=(16, 12))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.show()

# 4. Pairplot: hue shows the target diagnosis class
pair_features = [
    col for col in ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]
    if col in df.columns
]
if len(pair_features) >= 2:
    pair_df = df[pair_features + ["diagnosis"]].dropna()
    pair_grid = sns.pairplot(
        pair_df,
        hue="diagnosis",
        diag_kind="hist",
        corner=True
    )
    pair_grid.fig.suptitle("Pairplot of Selected Features by Diagnosis", y=1.02)
    pair_grid.savefig(OUTPUT_DIR / "pairplot_diagnosis.png", dpi=150)
    plt.show()

print(f"\nEDA charts saved in: {OUTPUT_DIR}")
