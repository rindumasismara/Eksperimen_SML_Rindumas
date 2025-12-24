import pandas as pd
import numpy as np
import os
import dagshub
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DAGSHUB_USERNAME = "rindumasismara"
REPO_NAME = "Eksperimen_SML_Rindumas"

DATASET_PATH = "onlinepaymentfraud_preprocessing.csv"
TARGET_COLUMN = "isFraud"
EXPERIMENT_NAME = "Fraud_RF_Tuning_Advanced"

def main():
    os.environ["DAGSHUB_DISABLE_OAUTH"] = "true"
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)

    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow")
    mlflow.set_experiment(EXPERIMENT_NAME)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH}")

    print("Loading dataset (sampled)...")
    df = pd.read_csv(DATASET_PATH, nrows=300_000)
    sample_size = min(200_000, len(df))
    df = df.sample(n=sample_size, random_state=42)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 150, "max_depth": 15},
    ]

    print(f"Mulai training {len(param_grid)} konfigurasi")

    for i, params in enumerate(param_grid):
        run_name = f"RF_Tuning_Run_{i+1}"

        with mlflow.start_run(run_name=run_name):
            print(f"Training {run_name}: {params}")
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_param("max_depth", params["max_depth"])

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, "model")

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix ({run_name})")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")

            cm_path = f"confusion_matrix_{i}.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)

            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]

            plt.figure(figsize=(8, 6))
            sns.barplot(
                x=importances[indices],
                y=X.columns[indices]
            )
            plt.title(f"Top 10 Feature Importance ({run_name})")

            fi_path = f"feature_importance_{i}.png"
            plt.savefig(fi_path)
            plt.close()
            mlflow.log_artifact(fi_path)

            os.remove(cm_path)
            os.remove(fi_path)

    print(f"Hasil ada di: https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}")

if __name__ == "__main__":
    main()
