import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

mlflow.set_experiment("Eksperimen_SML_Rindumas")

def main():
    df = pd.read_csv("onlinepaymentfraud_preprocessing.csv")

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = {
        "n_estimators": [10, 50],
        "max_depth": [5, 10]
    }

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    with mlflow.start_run():
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        # 🔹 manual logging
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        mlflow.sklearn.log_model(best_model, "model")

        print("Accuracy:", acc)

if __name__ == "__main__":
    main()
