import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("2018_Financial_Data.csv")

print("Dataset shape:", df.shape)


# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

# Prepare data

sector_counts = df["Sector"].value_counts()

sector_class_counts = pd.crosstab(
    df["Sector"],
    df["Class"]
)

sector_class_percent = pd.crosstab(
    df["Sector"],
    df["Class"],
    normalize="index"
) * 100


# Correlation analysis

correlations = df.corr(numeric_only=True)["2019 PRICE VAR [%]"]

correlations = correlations.drop(
    labels=["2019 PRICE VAR [%]", "Class"],
    errors="ignore"
)

top_correlations = correlations.loc[
    correlations.abs().nlargest(10).index
].sort_values()


# Create dashboard

fig, axes = plt.subplots(2, 2, figsize=(22, 18))


# Chart 1: Companies by Sector

sector_counts.plot(kind="bar", ax=axes[0, 0])

axes[0, 0].set_title("Companies by Sector")
axes[0, 0].set_xlabel("Sector")
axes[0, 0].set_ylabel("Number of Companies")
axes[0, 0].tick_params(axis="x", rotation=45)


# Chart 2: Class Distribution

sector_class_counts.plot(
    kind="bar",
    ax=axes[0, 1]
)

axes[0, 1].set_title("Class Distribution by Sector")
axes[0, 1].set_xlabel("Sector")
axes[0, 1].set_ylabel("Number of Companies")
axes[0, 1].tick_params(axis="x", rotation=45)


# Chart 3: Class Distribution (%)

sector_class_percent.plot(
    kind="bar",
    stacked=True,
    ax=axes[1, 0]
)

axes[1, 0].set_title("Class Distribution by Sector (%)")
axes[1, 0].set_xlabel("Sector")
axes[1, 0].set_ylabel("Percentage")
axes[1, 0].tick_params(axis="x", rotation=45)


# Chart 4: Top Correlations

top_correlations.plot(
    kind="barh",
    ax=axes[1, 1]
)

axes[1, 1].set_title(
    "Top 10 Financial Indicators Correlated with Stock Returns"
)
axes[1, 1].set_xlabel("Correlation")


# Final layout

plt.suptitle(
    "Financial Data Analysis Dashboard",
    fontsize=18
)

for ax in axes.flat:
    ax.tick_params(axis="x", labelsize=8)

plt.subplots_adjust(
    hspace=0.45,
    wspace=0.35,
    bottom=0.15
)

# Improve labels readability

for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
    ax.tick_params(axis="x", labelsize=8)

    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")

axes[1, 1].tick_params(axis="y", labelsize=8.5, pad=3)

fig.subplots_adjust(
    left=0.07,
    right=0.98,
    bottom=0.15,
    top=0.90,
    hspace=1.0,
    wspace=0.65
)

axes[0, 0].set_xlabel("")
axes[0, 1].set_xlabel("")
axes[1, 0].set_xlabel("")

plt.show()


# ==========================================
# 3. MACHINE LEARNING
# ==========================================

# Prepare data

ml_df = df.replace([np.inf, -np.inf], np.nan).copy()

ml_df = ml_df.dropna(subset=["Class"])

y = ml_df["Class"]

X = ml_df.drop(
    columns=["Class", "2019 PRICE VAR [%]", "Unnamed: 0"],
    errors="ignore"
)


# Remove columns with more than 50% missing values

X = X.loc[:, X.isnull().mean() <= 0.5]


# Identify column types

numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(exclude=np.number).columns


# Data preprocessing

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. CREATE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# 5. TRAIN AND EVALUATE
# ==========================================

results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    if name == "Random Forest":
        rf_pipeline = pipeline

    predictions = pipeline.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1 Score": f1_score(
            y_test,
            predictions,
            average="weighted"
        )
    })

    print("\n", "=" * 40)
    print(name)
    print("=" * 40)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("F1 Score:", f1_score(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# Baseline accuracy

print("\nBaseline accuracy:")
print(y_test.value_counts(normalize=True).max())


# Check potential future-data columns

print("\nPotential future-data columns:")

print([
    col for col in X.columns
    if "2019" in col.lower() or "price var" in col.lower()
])


# ==========================================
# 6. FEATURE IMPORTANCE
# ==========================================

rf_model = rf_pipeline.named_steps["model"]

importances = rf_model.feature_importances_

feature_names = (
    rf_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\nTop 15 Most Important Features:")
print(importance_df.head(15))


# Feature Importance Graph

top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 15 Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Financial Indicators")

plt.tight_layout()
plt.show()


# ==========================================
# 7. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.to_string(index=False))


print("\nPROJECT COMPLETED")