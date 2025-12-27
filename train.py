import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import mlflow
import mlflow.sklearn

# =========================
# Configuration MLflow
# =========================
mlflow.set_experiment("iris-dvc-mlops")

# =========================
# Charger le dataset
# =========================
df = pd.read_csv("data/iris.csv")

# Séparer features et target
X = df.drop(columns="target")
y = df["target"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Entraînement avec MLflow
# =========================
with mlflow.start_run():

    # Hyperparamètres
    n_estimators = 200
    random_state = 42

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    # Modèle
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)

    # Sauvegarde du modèle
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    mlflow.sklearn.log_model(model, "model")

    print(f"Modèle entraîné avec accuracy = {acc:.4f}")
    print("Modèle sauvegardé dans model.pkl")

