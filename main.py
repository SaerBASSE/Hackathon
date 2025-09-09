import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('data/waiting_times_train.csv')
df['date'] = pd.to_datetime(df['DATETIME'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month   
df['day'] = df['date'].dt.day
df['hours'] = df['date'].dt.hour
df['minutes'] = df['date'].dt.minute

df_encoded = pd.get_dummies(df, columns=['ENTITY_DESCRIPTION_SHORT'], drop_first=True)

X = df_encoded.drop(columns=['WAIT_TIME_IN_2H'])
X.fillna(100000000, inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA()
pca_full.fit(X_scaled)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
         pca_full.explained_variance_ratio_, 'bo-')
plt.xlabel('Composante')
plt.ylabel('Variance expliquée')
plt.title('Variance par composante')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
         np.cumsum(pca_full.explained_variance_ratio_), 'ro-')
plt.xlabel('Composante')
plt.ylabel('Variance cumulée')
plt.title('Variance cumulée')
plt.axhline(y=0.95, color='k', linestyle='--', label='95%')
plt.legend()

plt.tight_layout()
plt.show()
