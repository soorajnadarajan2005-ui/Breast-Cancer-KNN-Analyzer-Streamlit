# Breast Cancer KNN Analyzer — Streamlit

An interactive educational project that demonstrates Exploratory Data Analysis (EDA) and K-Nearest Neighbors (KNN) classification with a CSV dataset.

> **Safety note:** This is for educational machine-learning practice only. It is not a medical diagnostic tool.

## Project structure

```text
Breast_Cancer_KNN_Streamlit/
├── data/
│   └── breast_cancer.csv
├── notebooks/
│   └── eda.py
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

- `app.py`: Streamlit web interface for dataset overview, EDA, KNN evaluation, and test-record inspection.
- `main.py`: Standalone command-line KNN workflow.
- `notebooks/eda.py`: Standalone EDA script.
- `data/breast_cancer.csv`: Place the dataset here, or upload a CSV from the app sidebar.

## Setup in VS Code

Open the project folder in VS Code, then select **Terminal → New Terminal**.

Optional virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Streamlit app

Open a terminal in the project root (the folder containing `app.py`) and run:

```bash
python -m streamlit run app.py
```

Open the **Local URL** printed in the terminal, commonly `http://localhost:8501`.
Use this command for the web app; `python main.py` runs the standalone script
instead. If the default port is already in use, Streamlit will print a URL using
another port.

On Windows, if Streamlit is not installed in the selected Python environment,
install the project dependencies first:

```powershell
python -m pip install -r requirements.txt
```

## App features

- Upload a CSV or load `data/breast_cancer.csv` automatically.
- View dataset preview, shape, class counts, missing values, and descriptive statistics.
- Explore a feature distribution and correlation heatmap; generate the slower pairplot on demand.
- Adjust K, test-set percentage, and StandardScaler.
- View test accuracy, classification report, confusion matrix, and accuracy comparison across K values.
- Inspect predictions for individual held-out test records.

## Expected CSV format

The CSV must include a `diagnosis` column, typically with `B` and `M` labels. If present, the code ignores `id` and drops the empty `Unnamed: 32` column. Other feature columns should contain numerical values.

## KNN in brief

KNN calculates distances between records, selects the K nearest training examples, and predicts a class using majority voting. StandardScaler is useful because KNN depends on distances across numerical features.

## Run standalone scripts

```bash
python notebooks/eda.py
python main.py
```
