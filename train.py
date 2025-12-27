import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Charger le dataset
df = pd.read_csv("data/iris.csv")

# Séparer features et target
X = df.drop(columns="target")
y = df["target"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Sauvegarder le modèle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modèle entraîné et sauvegardé dans model.pkl")
