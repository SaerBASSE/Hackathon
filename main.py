import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Charger le fichier CSV
file_path = "waiting_times_train"
df = pd.read_csv(file_path)

# 1. Conversion de la colonne DATETIME en format datetime
df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")

# 2. Encodage des noms des attractions (ENTITY_DESCRIPTION_SHORT) en entiers
encoder = LabelEncoder()
df["ENTITY_ID"] = encoder.fit_transform(df["ENTITY_DESCRIPTION_SHORT"])

# 3. Suppression de la colonne texte d'origine
df = df.drop(columns=["ENTITY_DESCRIPTION_SHORT"])

# Aperçu des premières lignes
print(df.head())

