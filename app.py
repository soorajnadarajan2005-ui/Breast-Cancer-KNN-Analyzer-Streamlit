from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer.csv"

st.set_page_config(
    page_title="Breast Cancer KNN Analyzer",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Breast Cancer Classification")
st.caption("Interactive K-Nearest Neighbors (KNN) learning project")
st.warning(
    "Educational use only. This application is not a medical diagnostic tool "
    "and must not be used to make healthcare decisions."
)


@st.cache_data
def load_csv(file_bytes, file_name):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


def clean_dataset(data):
    data = data.copy()
    data = data.drop(columns=["Unnamed: 32"], errors="ignore")
    if "diagnosis" not in data.columns:
        raise ValueError(
            "The CSV needs a 'diagnosis' column with target labels such as B and M."
        )

    data["diagnosis"] = data["diagnosis"].astype(str).str.strip()
    X = data.drop(columns=["id", "diagnosis"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    y = data["diagnosis"]

    valid = X.notna().all(axis=1) & y.notna() & y.ne("")
    cleaned = data.loc[valid].copy()
    X = X.loc[valid]
    y = y.loc[valid]

    if X.empty:
        raise ValueError("No complete numeric feature rows were found in this CSV.")
    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")
    return cleaned, X, y


with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
    k_value = st.slider("Number of neighbors (K)", min_value=1, max_value=15, value=5)
    test_size = st.slider(
        "Test data percentage",
        min_value=10,
        max_value=40,
        value=30,
        step=5,
    )
    scale_features = st.checkbox("Apply StandardScaler", value=True)
    st.caption("Scaling is recommended for distance-based KNN.")

try:
    if uploaded_file is not None:
        raw_df = load_csv(uploaded_file.getvalue(), uploaded_file.name)
        source_name = uploaded_file.name
    elif DEFAULT_DATA_PATH.exists():
        raw_df = pd.read_csv(DEFAULT_DATA_PATH)
        source_name = "data/breast_cancer.csv"
    else:
        st.info(
            "Upload breast_cancer.csv using the sidebar, or place it at "
            "`data/breast_cancer.csv` in the project folder."
        )
        st.stop()

    df, X, y = clean_dataset(raw_df)
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

st.caption(f"Dataset source: `{source_name}`")

tab_overview, tab_eda, tab_model, tab_predict = st.tabs(
    ["📊 Dataset Overview", "🔎 EDA", "🤖 KNN Model", "🧪 Sample Prediction"]
)

with tab_overview:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Valid records", f"{len(df):,}")
    c2.metric("Numerical features", f"{X.shape[1]}")
    c3.metric("Diagnosis classes", f"{y.nunique()}")

    st.write("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.write("Diagnosis class counts")
        counts = y.value_counts().rename_axis("Diagnosis").reset_index(name="Count")
        st.bar_chart(counts.set_index("Diagnosis"))
    with right:
        st.write("Missing values in the original data")
        missing = raw_df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values detected in the uploaded dataset.")
        else:
            st.dataframe(
                missing.rename("Missing values").to_frame(),
                use_container_width=True,
            )

    st.write("Numerical summary")
    st.dataframe(X.describe().T, use_container_width=True)

with tab_eda:

    st.subheader("Count Plot - Diagnosis Distribution")
    fig_count, ax_count = plt.subplots(figsize=(7, 4))
    sns.countplot(
        data=df,
        x="diagnosis",
        hue="diagnosis",
        palette="Set2",
        legend=False,
        ax=ax_count,
    )
    ax_count.set_title("Count of Benign and Malignant Cases")
    ax_count.set_xlabel("Diagnosis")
    ax_count.set_ylabel("Count")
    st.pyplot(fig_count)
    plt.close(fig_count)

    st.subheader("Exploratory data analysis")

    numeric_cols = X.columns.tolist()
    default_feature = "radius_mean" if "radius_mean" in numeric_cols else numeric_cols[0]
    chosen_feature = st.selectbox(
        "Choose a feature to inspect",
        numeric_cols,
        index=numeric_cols.index(default_feature),
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=pd.DataFrame({chosen_feature: X[chosen_feature], "diagnosis": y}),
        x=chosen_feature,
        hue="diagnosis",
        kde=True,
        ax=ax,
    )
    ax.set_title(f"{chosen_feature} distribution by diagnosis")
    st.pyplot(fig)
    plt.close(fig)

    st.write("Correlation heatmap (first 15 numerical features)")
    corr_cols = numeric_cols[:15]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(X[corr_cols].corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlations")
    st.pyplot(fig)
    plt.close(fig)

    pair_candidates = [
        c for c in ["radius_mean", "texture_mean", "perimeter_mean", "area_mean"]
        if c in numeric_cols
    ]
    if len(pair_candidates) >= 2:
        show_pairplot = st.checkbox(
            "Show pairplot (takes longer to generate)",
            value=False,
        )
        if show_pairplot:
            with st.spinner("Creating pairplot..."):
                pair_df = X[pair_candidates].copy()
                pair_df["diagnosis"] = y.values
                pair_grid = sns.pairplot(
                    pair_df,
                    hue="diagnosis",
                    diag_kind="hist",
                    corner=True,
                )
                pair_grid.fig.suptitle("Selected Features by Diagnosis", y=1.02)
                st.pyplot(pair_grid.fig)
                plt.close(pair_grid.fig)

with tab_model:
    st.subheader("Train and evaluate KNN")
    st.write(
        "The data is split into training and testing sets. If scaling is enabled, "
        "the scaler is fitted on training data only to avoid data leakage."
    )

    class_counts = y.value_counts()
    can_stratify = class_counts.min() >= 2
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=42,
            stratify=y if can_stratify else None,
        )

        if k_value > len(X_train):
            st.error(
                f"K={k_value} is larger than the training set "
                f"({len(X_train)} rows). Choose a smaller K."
            )
            st.stop()

        if scale_features:
            scaler = StandardScaler()
            X_train_used = scaler.fit_transform(X_train)
            X_test_used = scaler.transform(X_test)
        else:
            X_train_used = X_train
            X_test_used = X_test

        model = KNeighborsClassifier(n_neighbors=k_value)
        model.fit(X_train_used, y_train)
        y_pred = model.predict(X_test_used)

        accuracy = accuracy_score(y_test, y_pred)
        m1, m2, m3 = st.columns(3)
        m1.metric("Test accuracy", f"{accuracy:.2%}")
        m2.metric("Training records", f"{len(X_train):,}")
        m3.metric("Testing records", f"{len(X_test):,}")

        st.write("Classification report")
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)

        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

        st.write("Compare different K values")
        comparison = []
        for k in [1, 3, 5, 7, 9, 11, 13, 15]:
            if k <= len(X_train):
                candidate = KNeighborsClassifier(n_neighbors=k)
                candidate.fit(X_train_used, y_train)
                candidate_pred = candidate.predict(X_test_used)
                comparison.append(
                    {"K": k, "Accuracy": accuracy_score(y_test, candidate_pred)}
                )

        comparison_df = pd.DataFrame(comparison)
        st.line_chart(comparison_df.set_index("K"))
        st.dataframe(comparison_df, use_container_width=True)

        # Keep objects for sample prediction in the session.
        st.session_state["knn_model"] = model
        st.session_state["knn_scaler"] = scaler if scale_features else None
        st.session_state["knn_X_test"] = X_test
        st.session_state["knn_y_test"] = y_test
        st.session_state["knn_predictions"] = pd.Series(
            y_pred, index=X_test.index, name="Predicted diagnosis"
        )
        st.session_state["knn_feature_names"] = X.columns.tolist()
        st.session_state["knn_scaled"] = scale_features

    except ValueError as exc:
        st.error(f"Could not train the model: {exc}")

with tab_predict:
    st.subheader("Inspect a held-out test record")
    st.write(
        "Choose a record from the held-out test set to compare its actual label "
        "with the model's prediction. This is a demonstration, not a medical assessment."
    )

    if "knn_predictions" not in st.session_state:
        st.info("Open the KNN Model tab first to train the model.")
    else:
        predictions = st.session_state["knn_predictions"]
        test_X = st.session_state["knn_X_test"]
        test_y = st.session_state["knn_y_test"]
        selected_index = st.selectbox(
            "Select test record",
            options=list(test_X.index),
            format_func=lambda idx: f"Record index {idx}",
        )

        actual = test_y.loc[selected_index]
        predicted = predictions.loc[selected_index]
        p1, p2 = st.columns(2)
        p1.metric("Actual class", str(actual))
        p2.metric("Predicted class", str(predicted))

        if actual == predicted:
            st.success("The prediction matches this record's dataset label.")
        else:
            st.error("The prediction does not match this record's dataset label.")

        st.write("Feature values for this record")
        st.dataframe(
            test_X.loc[[selected_index]].T.rename(columns={selected_index: "Value"}),
            use_container_width=True,
        )
