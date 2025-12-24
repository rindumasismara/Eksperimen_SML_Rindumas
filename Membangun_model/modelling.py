import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Fraud Detection Basic")

def main():
    df = pd.read_csv("onlinepaymentfraud_preprocessing.csv")

    if "amount_bin" in df.columns:
        df = df.drop(columns=["amount_bin"])

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        mlflow.sklearn.autolog()

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy:", acc)
        print("F1 Score:", f1)

if __name__ == "__main__":
    main()
