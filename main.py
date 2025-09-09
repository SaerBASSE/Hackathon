import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Charger le fichier CSV
file_path = "ton_fichier.csv"
df = pd.read_csv(file_path)

# 1. Conversion de la colonne DATETIME en format datetime
df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")

# 2. Création de nouvelles colonnes temporelles
df["YEAR"] = df["DATETIME"].dt.year
df["MONTH"] = df["DATETIME"].dt.month
df["DAY"] = df["DATETIME"].dt.day
df["HOUR"] = df["DATETIME"].dt.hour
df["MINUTE"] = df["DATETIME"].dt.minute
df["DAY_OF_WEEK"] = df["DATETIME"].dt.dayofweek  # 0=lundi, 6=dimanche

# 3. Encodage des attractions
encoder = LabelEncoder()
df["ENTITY_ID"] = encoder.fit_transform(df["ENTITY_DESCRIPTION_SHORT"])

# 4. Suppression des colonnes inutiles (ancienne date et texte)
df = df.drop(columns=["DATETIME", "ENTITY_DESCRIPTION_SHORT"])

# Résultat
print(df.head())
