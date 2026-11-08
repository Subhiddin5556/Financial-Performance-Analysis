import pandas as pd
import numpy as np

# Load training data
train_data = []

for year in [2014, 2015, 2016, 2017]:
    df = pd.read_csv(f"{year}_Financial_Data.csv")

    df = df.drop(
        columns=["Class", f"{year + 1} PRICE VAR [%]", "Unnamed: 0"],
        errors="ignore"
    )

    train_data.append(df)

X_train = pd.concat(train_data, ignore_index=True)

# Load test data
test_df = pd.read_csv("2018_Financial_Data.csv")

y_train = pd.concat([
    pd.read_csv(f"{year}_Financial_Data.csv")["Class"]
    for year in [2014, 2015, 2016, 2017]
], ignore_index=True)

y_test = test_df["Class"]

X_test = test_df.drop(
    columns=["Class", "2019 PRICE VAR [%]", "Unnamed: 0"],
    errors="ignore"
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Handle missing and infinite values
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Identify column types
numeric_features = X_train.select_dtypes(include=np.number).columns
categorical_features = X_train.select_dtypes(exclude=np.number).columns

# Preprocessing
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

# Models
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

# Train and evaluate
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print("\n" + "=" * 40)
    print(name)
    print("=" * 40)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, average="weighted"))
    print("Recall:", recall_score(y_test, predictions, average="weighted"))
    print("F1 Score:", f1_score(y_test, predictions, average="weighted"))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
print("\nBaseline Accuracy:")
print(y_test.value_counts(normalize=True).max())