import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# === 1. Charger et préparer les données ===
file_path = "/workspaces/Hackathon/data/waiting_times_train.csv"
df = pd.read_csv(file_path)

# Conversion de la date
df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
df["YEAR"] = df["DATETIME"].dt.year
df["MONTH"] = df["DATETIME"].dt.month
df["DAY"] = df["DATETIME"].dt.day
df["HOUR"] = df["DATETIME"].dt.hour
df["DAY_OF_WEEK"] = df["DATETIME"].dt.dayofweek

# Encodage des attractions
encoder = LabelEncoder()
df["ENTITY_ID"] = encoder.fit_transform(df["ENTITY_DESCRIPTION_SHORT"])

# Gestion des colonnes parade et night show
for col in ["TIME_TO_PARADE_1", "TIME_TO_PARADE_2", "TIME_TO_NIGHT_SHOW"]:
    df[col + "_AVAILABLE"] = df[col].notna().astype(int)
    df[col] = df[col].fillna(-1)

# Supprimer colonnes inutiles
df = df.drop(columns=["DATETIME", "ENTITY_DESCRIPTION_SHORT"])

# === 2. Standardisation ===
features = df.drop(columns=["WAIT_TIME_IN_2H"])  # X
target = df["WAIT_TIME_IN_2H"]  # y

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# === 3. PCA ===
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_

# === 4. PLots ===

# (a) Variance expliquée
plt.figure(figsize=(8,5))
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7, label="Composantes")
plt.step(range(1, len(explained_variance)+1), np.cumsum(explained_variance), where="mid", label="Cumulée")
plt.xlabel("Numéro de la composante")
plt.ylabel("Variance expliquée")
plt.title("Variance expliquée par les composantes PCA")
plt.legend()
plt.show()

# (b) Projection sur les 2 premières composantes
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=target, cmap="viridis", s=10)
plt.colorbar(label="Temps d'attente dans 2h")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Projection PCA (2 premières composantes)")
plt.show()

# (c) Contribution des variables (loadings)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(features.columns))],
    index=features.columns
)

plt.figure(figsize=(10,6))
sns.heatmap(loadings.iloc[:,:5], cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Contribution des variables aux 5 premières composantes")
plt.show()
