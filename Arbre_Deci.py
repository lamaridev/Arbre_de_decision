import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Charger les données
data = pd.read_csv("C:/Users/hp/anaconda3/salaries.csv")

# Séparer les caractéristiques (X) et les étiquettes (y)
X = data[["company", "job", "degree"]]
y = data["salary_more_then_100k"]

# Créer des instances de LabelEncoder pour chaque colonne
label_encoders = {}
for column in X.columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])
    label_encoders[column] = encoder

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle d'arbre de décision
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
predictions = model.predict(X_test)

# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


# Créer un dictionnaire avec les valeurs pour le cas à tester
sample_input = {
    "company": "abc pharma",
    "job": "business manager",
    "degree": "bachelaors"
}

# Convertir les valeurs en valeurs encodées
sample_input_encoded = {}
for column, value in sample_input.items():
    sample_input_encoded[column] = label_encoders[column].transform([value])[0]

# Faire une prédiction pour le cas donné
prediction = model.predict([list(sample_input_encoded.values())])

# Décoder la prédiction en une étiquette compréhensible
decoded_prediction = "Yes" if prediction[0] == 1 else "No"

# Afficher la prédiction
print(f"Prediction for the given case: {decoded_prediction}")

