import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def main():
    mlflow.set_experiment("Fraud Detection - Tuning")

    df = pd.read_csv("onlinepaymentfraud_preprocessing.csv")

    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10]
    }

    model = RandomForestClassifier(random_state=42)

    with mlflow.start_run():
        grid = GridSearchCV(model, param_grid, cv=3, scoring="f1")
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.sklearn.log_model(best_model, "model")

if __name__ == "__main__":
    main()
