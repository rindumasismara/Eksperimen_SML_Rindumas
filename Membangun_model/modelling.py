import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = "onlinepaymentfraud_preprocessing.csv"
TARGET_COLUMN = "isFraud"

EXPERIMENT_NAME = "Eksperimen_SML_Rindumas"
RUN_NAME = "RandomForest_Basic"

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.autolog()

    with mlflow.start_run(run_name=RUN_NAME):

        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH}")

        print("Loading dataset...")
        df = pd.read_csv(DATASET_PATH)
        df = df.sample(n=200_000, random_state=42)

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Training model:")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        print("Evaluating model...")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        if os.path.exists(cm_path):
            os.remove(cm_path)

if __name__ == "__main__":
    main()
