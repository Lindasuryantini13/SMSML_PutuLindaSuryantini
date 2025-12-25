"""
Hyperparameter Tuning with MLflow Tracking
Author: Nama Siswa
Purpose: Kriteria 2 - MSML Submission
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer
)
import argparse
import os


def load_preprocessed_data(filepath):
    """
    Load preprocessed data

    Args:
        filepath: Path to preprocessed CSV file

    Returns:
        DataFrame: Loaded preprocessed data
    """
    print(f"Loading preprocessed data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets

    Args:
        df: Input DataFrame
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"\nSplitting data (test_size={test_size})...")

    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def tune_random_forest_grid(X_train, y_train, X_test, y_test):
    """
    Tune Random Forest using GridSearchCV

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
    """
    print("\n" + "="*60)
    print("Random Forest Hyperparameter Tuning - GridSearchCV")
    print("="*60)

    with mlflow.start_run(run_name="rf_grid_search"):
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        print("\nParameter Grid:")
        for key, value in param_grid.items():
            print(f"  {key}: {value}")

        # Create base model
        base_model = RandomForestClassifier(random_state=42)

        # GridSearchCV
        print("\nPerforming GridSearchCV (this may take a while)...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        # Fit
        grid_search.fit(X_train, y_train)

        # Best parameters
        best_params = grid_search.best_params_
        print("\nBest Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        # Log best parameters
        mlflow.log_params(best_params)
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("cv_folds", 5)

        # Best model
        best_model = grid_search.best_estimator_

        # Predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test),
            'test_recall': recall_score(y_test, y_pred_test),
            'test_f1': f1_score(y_test, y_pred_test),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba_test),
            'best_cv_score': grid_search.best_score_
        }

        # Log metrics
        mlflow.log_metrics(metrics)
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Log model
        mlflow.sklearn.log_model(best_model, "model")

        # Save grid search results
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_path = 'grid_search_results_rf.csv'
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        print("\nGridSearchCV completed and logged to MLflow")
        print("="*60)


def tune_random_forest_random(X_train, y_train, X_test, y_test):
    """
    Tune Random Forest using RandomizedSearchCV

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
    """
    print("\n" + "="*60)
    print("Random Forest Hyperparameter Tuning - RandomizedSearchCV")
    print("="*60)

    with mlflow.start_run(run_name="rf_random_search"):
        # Define parameter distributions
        param_distributions = {
            'n_estimators': [50, 75, 100, 125, 150, 175, 200],
            'max_depth': [5, 8, 10, 12, 15, 20, None],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 6, 8],
            'max_features': ['sqrt', 'log2', None]
        }

        print("\nParameter Distributions:")
        for key, value in param_distributions.items():
            print(f"  {key}: {value}")

        # Create base model
        base_model = RandomForestClassifier(random_state=42)

        # RandomizedSearchCV
        print("\nPerforming RandomizedSearchCV...")
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=20,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        # Fit
        random_search.fit(X_train, y_train)

        # Best parameters
        best_params = random_search.best_params_
        print("\nBest Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        # Log best parameters
        mlflow.log_params(best_params)
        mlflow.log_param("tuning_method", "RandomizedSearchCV")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("n_iter", 20)

        # Best model
        best_model = random_search.best_estimator_

        # Predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test),
            'test_recall': recall_score(y_test, y_pred_test),
            'test_f1': f1_score(y_test, y_pred_test),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba_test),
            'best_cv_score': random_search.best_score_
        }

        # Log metrics
        mlflow.log_metrics(metrics)
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Log model
        mlflow.sklearn.log_model(best_model, "model")

        # Save random search results
        results_df = pd.DataFrame(random_search.cv_results_)
        results_path = 'random_search_results_rf.csv'
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        print("\nRandomizedSearchCV completed and logged to MLflow")
        print("="*60)


def tune_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Tune Logistic Regression using GridSearchCV

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
    """
    print("\n" + "="*60)
    print("Logistic Regression Hyperparameter Tuning - GridSearchCV")
    print("="*60)

    with mlflow.start_run(run_name="lr_grid_search"):
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000, 2000]
        }

        print("\nParameter Grid:")
        for key, value in param_grid.items():
            print(f"  {key}: {value}")

        # Create base model
        base_model = LogisticRegression(random_state=42)

        # GridSearchCV
        print("\nPerforming GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        # Fit
        grid_search.fit(X_train, y_train)

        # Best parameters
        best_params = grid_search.best_params_
        print("\nBest Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        # Log best parameters
        mlflow.log_params(best_params)
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("cv_folds", 5)

        # Best model
        best_model = grid_search.best_estimator_

        # Predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test),
            'test_recall': recall_score(y_test, y_pred_test),
            'test_f1': f1_score(y_test, y_pred_test),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba_test),
            'best_cv_score': grid_search.best_score_
        }

        # Log metrics
        mlflow.log_metrics(metrics)
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Log model
        mlflow.sklearn.log_model(best_model, "model")

        # Save grid search results
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_path = 'grid_search_results_lr.csv'
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        print("\nGridSearchCV completed and logged to MLflow")
        print("="*60)


def main(data_path, experiment_name="telco-churn-tuning", method="all"):
    """
    Main hyperparameter tuning pipeline

    Args:
        data_path: Path to preprocessed data
        experiment_name: MLflow experiment name
        method: Tuning method (grid, random, or all)
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment: {experiment_name}")

    # Load data
    df = load_preprocessed_data(data_path)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n" + "="*60)
    print("STARTING HYPERPARAMETER TUNING")
    print("="*60)

    # Perform tuning based on method
    if method == "grid" or method == "all":
        tune_random_forest_grid(X_train, y_train, X_test, y_test)
        tune_logistic_regression(X_train, y_train, X_test, y_test)

    if method == "random" or method == "all":
        tune_random_forest_random(X_train, y_train, X_test, y_test)

    print("\n" + "="*60)
    print("ALL TUNING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nMLflow UI: mlflow ui")
    print(f"Then open: http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with MLflow tracking')
    parser.add_argument('--data', type=str, default='preprocessed_data.csv',
                        help='Path to preprocessed data')
    parser.add_argument('--experiment', type=str, default='telco-churn-tuning',
                        help='MLflow experiment name')
    parser.add_argument('--method', type=str, default='all',
                        choices=['grid', 'random', 'all'],
                        help='Tuning method: grid, random, or all')

    args = parser.parse_args()

    main(data_path=args.data, experiment_name=args.experiment, method=args.method)
