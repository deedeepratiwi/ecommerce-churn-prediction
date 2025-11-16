#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import pickle

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

from src.dict_vectorizer_transformer import DictVectorizerTransformer

# ----------------------------
# Paths
# ----------------------------
DATA_PATH = "data/processed_data.csv"
OUTPUT_PATH = "models/best_model.pkl"

# ----------------------------
# Columns
# ----------------------------
numerical = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 
                    'NumberOfAddress', 'OrderAmountHikeFromlastYear',
                    'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']

categorical = ['PreferredLoginDevice', 'CityTier', 'PreferredPaymentMode', 'Gender', 
                    'NumberOfDeviceRegistered', 'PreferedOrderCat', 
                    'SatisfactionScore', 'MaritalStatus', 'Complain']


target = 'Churn'
random_seed = 42
n_splits = 5

# ----------------------------
# Model search
# ----------------------------
models = {
    "LogisticRegression": GridSearchCV(
        estimator=LogisticRegression(max_iter=10000, random_state=random_seed),
        param_grid={
            "C": [0.001, 0.01, 0.05, 0.08, 0.2],
            "penalty": ["l2"],
            "solver": ["lbfgs"]
        },
        cv=5,
        scoring="f1",
        n_jobs=-1
    ),

    "DecisionTree": GridSearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid={
            "max_depth": [3, 4, 5, 6, 10, 15],
            "min_samples_leaf": [1, 5, 10, 20, 50, 100]
        },
        cv=5,
        scoring="f1",
        n_jobs=-1
    ),

    "RandomForest": GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_seed),
        param_grid={
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, 20],
            "min_samples_leaf": [1, 3, 5, 10, 50]
        },
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
}

# ----------------------------
# Load data
# ----------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)

    return df

# ----------------------------
# Split data
# ----------------------------
def split_data(df, target_col="Churn"):
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=random_seed, stratify=df[target]
    )

    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=random_seed, stratify=df_full_train[target],
    )

    X_train = df_train.drop(columns=[target])
    y_train = df_train[target].values

    X_val = df_val.drop(columns=[target])
    y_val = df_val[target].values

    X_test = df_test.drop(columns=[target])
    y_test = df_test[target].values

    return X_train, y_train, X_val, y_val, X_test, y_test

# ----------------------------
# Build pipelines
# ----------------------------
def build_pipelines():
    pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([
            ('dictvec', DictVectorizerTransformer()),
            ('model', model)
        ])
        pipelines[name] = pipe
    
    return pipelines

# ----------------------------
# Manual Cross Validation
# ----------------------------
def cross_validation(pipe, X, y, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train_kf = X.iloc[train_idx]
        y_train_kf = y[train_idx]

        X_val_kf = X.iloc[val_idx]
        y_val_kf = y[val_idx]

        pipe.fit(X_train_kf, y_train_kf)
        y_pred = pipe.predict(X_val_kf)
        scores.append(f1_score(y_val_kf, y_pred))

    return np.mean(scores)

# ----------------------------
# Train & MLflow logging
# ----------------------------
def train_best_model(X_train, X_val, X_test, y_train, y_val, y_test):
    pipelines = build_pipelines()
    results = {}

    # MLflow config
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ecommerce-churn")

    for name, pipe in pipelines.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            f1 = cross_validation(pipe, pd.concat([X_train, X_val]), np.concatenate([y_train, y_val]), n_splits)
            results[name] = f1
            print(f"{name} | F1-score: {f1:.4f}\n")

            # Log metric
            mlflow.log_metric("cv_f1", f1)

            # Log best hyperparams after fitting GridSearchCV inside cross-validation
            best_est = pipe.named_steps["model"].best_estimator_
            mlflow.log_params(best_est.get_params())

            # Log pipeline object
            mlflow.sklearn.log_model(pipe, name="pipeline")        

    # Pick best model
    best_model_name = max(results, key=results.get)
    best_pipeline = pipelines[best_model_name]

    print("====================================")
    print(f" Best model: {best_model_name} (F1={results[best_model_name]:.4f})")
    print("====================================")

    # Train on full dataset
    X_full = pd.concat([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    best_pipeline.fit(X_full, y_full)

    # Evaluate on test set
    y_pred_test = best_pipeline.predict(X_test)
    f1_test = f1_score(y_test, y_pred_test)

    print(f"\nTest F1 Score ({best_model_name}): {f1_test:.4f}")

    # Track test score
    with mlflow.start_run(run_name=f"{best_model_name}-final") as run:
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("test_f1", f1_test)

        # Log and register the model
        mlflow.sklearn.log_model(
            sk_model=best_pipeline, 
            name="final_model", 
            registered_model_name="ecommerce-churn-model"
        )

    # Set stage and tags
    client = MlflowClient()
    model_uri = f"runs:/{run.info.run_id}/final_model"
    model_version = client.get_latest_versions("ecommerce-churn-model", stages=["None"])[-1].version

    client.transition_model_version_stage(
        name="ecommerce-churn-model",
        version=model_version,
        stage="Production",
        archive_existing_versions=True
    )

    client.set_model_version_tag(
        name="ecommerce-churn-model",
        version=model_version,
        key="valid_from",
        value=str(pd.Timestamp.now().date())
    )
    client.set_model_version_tag(
        name="ecommerce-churn-model",
        version=model_version,
        key="model_type",
        value=best_model_name
    )

    return best_pipeline, best_model_name

# ----------------------------
# Save pickle
# ----------------------------
def save_model(pipeline, model_name, output_path=OUTPUT_PATH):
    with open(output_path, "wb") as f:
        pickle.dump({
            "model_name": model_name,
            "pipeline": pipeline
        }, f)

    print(f"Saved model to {output_path}")

def main():
    df = load_data()

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

    pipeline, best_name = train_best_model(X_train, X_val, X_test, y_train, y_val, y_test)

    save_model(pipeline, best_name, OUTPUT_PATH)

if __name__ == "__main__":
    main()