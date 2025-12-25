"""
Model Training with MLflow Tracking
Author: Nama Siswa
Purpose: Kriteria 2 - MSML Submission
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_confusion_matrix(y_true, y_pred, output_path):
    """
    Plot and save confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Confusion matrix saved to {output_path}")


def plot_feature_importance(model, feature_names, output_path, top_n=15):
    """
    Plot and save feature importance (for tree-based models)

    Args:
        model: Trained model
        feature_names: List of feature names
        output_path: Path to save the plot
        top_n: Number of top features to display
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importances')
        plt.barh(range(top_n), importances[indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"Feature importance plot saved to {output_path}")


def train_random_forest(X_train, y_train, X_test, y_test, run_name="random_forest"):
    """
    Train Random Forest model with MLflow tracking

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        run_name: MLflow run name

    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("Training Random Forest Classifier")
    print("="*60)

    with mlflow.start_run(run_name=run_name):
        # Set parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }

        # Log parameters
        mlflow.log_params(params)
        print("\nParameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        # Train model
        print("\nTraining model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test),
            'test_recall': recall_score(y_test, y_pred_test),
            'test_f1': f1_score(y_test, y_pred_test),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba_test)
        }

        # Log metrics
        mlflow.log_metrics(metrics)
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Create artifacts directory
        os.makedirs('artifacts', exist_ok=True)

        # Plot and log confusion matrix
        cm_path = 'artifacts/confusion_matrix_rf.png'
        plot_confusion_matrix(y_test, y_pred_test, cm_path)
        mlflow.log_artifact(cm_path)

        # Plot and log feature importance
        fi_path = 'artifacts/feature_importance_rf.png'
        plot_feature_importance(model, X_train.columns.tolist(), fi_path)
        mlflow.log_artifact(fi_path)

        # Log classification report
        report = classification_report(y_test, y_pred_test, target_names=['No Churn', 'Churn'])
        print("\nClassification Report:")
        print(report)

        report_path = 'artifacts/classification_report_rf.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Log model
        mlflow.sklearn.log_model(model, "model")
        print("\nModel logged to MLflow")

        print("\n" + "="*60)
        print("Random Forest Training Completed")
        print("="*60)

        return model


def train_logistic_regression(X_train, y_train, X_test, y_test, run_name="logistic_regression"):
    """
    Train Logistic Regression model with MLflow tracking

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        run_name: MLflow run name

    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("Training Logistic Regression")
    print("="*60)

    with mlflow.start_run(run_name=run_name):
        # Set parameters
        params = {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs'
        }

        # Log parameters
        mlflow.log_params(params)
        print("\nParameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        # Train model
        print("\nTraining model...")
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test),
            'test_recall': recall_score(y_test, y_pred_test),
            'test_f1': f1_score(y_test, y_pred_test),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba_test)
        }

        # Log metrics
        mlflow.log_metrics(metrics)
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Create artifacts directory
        os.makedirs('artifacts', exist_ok=True)

        # Plot and log confusion matrix
        cm_path = 'artifacts/confusion_matrix_lr.png'
        plot_confusion_matrix(y_test, y_pred_test, cm_path)
        mlflow.log_artifact(cm_path)

        # Log classification report
        report = classification_report(y_test, y_pred_test, target_names=['No Churn', 'Churn'])
        print("\nClassification Report:")
        print(report)

        report_path = 'artifacts/classification_report_lr.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Log model
        mlflow.sklearn.log_model(model, "model")
        print("\nModel logged to MLflow")

        print("\n" + "="*60)
        print("Logistic Regression Training Completed")
        print("="*60)

        return model


def main(data_path, experiment_name="telco-churn-classification"):
    """
    Main training pipeline

    Args:
        data_path: Path to preprocessed data
        experiment_name: MLflow experiment name
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment: {experiment_name}")

    # Load data
    df = load_preprocessed_data(data_path)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train models
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING")
    print("="*60)

    # Random Forest
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)

    # Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train, X_test, y_test)

    print("\n" + "="*60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("="*60)
    print(f"\nMLflow UI: mlflow ui")
    print(f"Then open: http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train models with MLflow tracking')
    parser.add_argument('--data', type=str, default='preprocessed_data.csv',
                        help='Path to preprocessed data')
    parser.add_argument('--experiment', type=str, default='telco-churn-classification',
                        help='MLflow experiment name')

    args = parser.parse_args()

    main(data_path=args.data, experiment_name=args.experiment)
