# src/preprocessing.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os

def load_and_preprocess():
    # Load dataset
    df = pd.read_csv("data/heart.csv")

    # Convert target to binary (0 = no heart disease, 1 = heart disease)
    df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

    # Drop irrelevant columns
    X = df.drop(columns=["id", "num", "dataset"])  # dataset removed
    y = df["num"]

    # Define feature types
    num_features = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    cat_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

    # Numeric preprocessing: imputation + scaling
    num_pipeline = Pipeline([
        ("imputer", IterativeImputer(random_state=42)),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing: mode imputation + one-hot encoding
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    return X, y, preprocessor