# src/train_model.py

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from preprocessing import load_and_preprocess
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load data and preprocessor
X, y, preprocessor = load_and_preprocess()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
}

# Define hyperparameter grids
params = {
    "Logistic Regression": {"model__C": np.logspace(-3, 3, 10)},
    "Random Forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    },
    "XGBoost": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0]
    }
}

best_auc = 0
best_model = None

# Train and tune models
for name, model in models.items():
    print(f"Training {name}...")
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    search = RandomizedSearchCV(
        pipe,
        param_distributions=params[name],
        n_iter=10,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)
    y_prob = search.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"{name} AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model = search.best_estimator_

# Save the best model
joblib.dump(best_model, "models/best_model.pkl")
print(f"✅ Best model saved with ROC-AUC: {best_auc:.4f}")
