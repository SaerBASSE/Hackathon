import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import multiprocessing
import optuna
import warnings
warnings.filterwarnings("ignore")

class XGBoostOptimize:
    def __init__(self,X,y, n_trials=100, cv_folds=5):
        self.X = X
        self.y = y 
        
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_model = None
        self.study = None
        
        
    def objective(self, trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step = 50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3,log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'tree_method': 'hist',
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 0,
            'objective': 'reg:squarederror'
        }
        
        model = xgb.XGBRegressor(**param)
        scores = cross_val_score(
            model, self.X, self.y, 
            cv=self.cv_folds, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        return -np.mean(scores)
    def optimize(self,timeout = 3600):
        print("debut de l'optimisation")
        self.study = optuna.create_study(direction = 'minimize',sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20))
        def callback(study, trial):
            if trial.number % 20 == 0:
                print(f"Trial {trial.number}: Best RMSE so far: {study.best_value:.6f}")
        self.study.optimize(self.objective, n_trials=self.n_trials, timeout=timeout, callbacks=[callback], show_progress_bar=True)
        print("Optimisation terminée")
        return self.study.best_params
    def get_best_model(self):
        if self.study is None:
            raise ValueError("You need to run the optimize method before getting the best model.")
        best_params = self.study.best_params
        best_params.update({
            'tree_method': 'hist',
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 0
        })
        self.best_model = xgb.XGBRegressor(**best_params)
        self.best_model.fit(self.X, self.y)
        
        
    def predict(self, X):
        return self.best_model.predict(X)
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
df1 = pd.read_csv('/Users/home/Documents/Hackathon/data/waiting_times_train.csv')
df2 = pd.read_csv('/Users/home/Documents/Hackathon/data/weather_data.csv')
df2.fillna(0,inplace=True)
df = pd.merge(df1, df2, on='DATETIME', how='inner')
df['date'] = pd.to_datetime(df['DATETIME'],format='%Y-%m-%d %H:%M:%S', errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hours'] = df['date'].dt.hour
df['minutes'] = df['date'].dt.minute

df_encoded = pd.get_dummies(df, columns=['ENTITY_DESCRIPTION_SHORT'], drop_first=True)
Features= ['year', 'month', 'day', 'hours', 'minutes','temp','dew_point','feels_like','pressure','humidity','wind_speed','rain_1h','snow_1h','clouds_all'] + [col for col in df_encoded.columns if col.startswith('ENTITY_DESCRIPTION_SHORT_')]
X = df_encoded[Features]

X.fillna(10000000000000, inplace=True)
scaler = StandardScaler()


X_valid = pd.read_csv('/Users/home/Documents/Hackathon/data/waiting_times_X_test_val.csv')
X_valid = pd.merge(X_valid, df2, on='DATETIME', how='inner')
X_valid['date'] = pd.to_datetime(X_valid['DATETIME'],format='%Y-%m-%d %H:%M:%S', errors='coerce')
X_valid['year'] = X_valid['date'].dt.year
X_valid['month'] = X_valid['date'].dt.month
X_valid['day'] = X_valid['date'].dt.day 
X_valid['hours'] = X_valid['date'].dt.hour
X_valid['minutes'] = X_valid['date'].dt.minute
X_valid_encoded = pd.get_dummies(X_valid, columns=['ENTITY_DESCRIPTION_SHORT'], drop_first=True)
X_valid_encoded.fillna(10000000000000, inplace=True)
X_valid_final = X_valid_encoded.reindex(columns=X.columns, fill_value=0)


model = XGBoostOptimize(X, df['WAIT_TIME_IN_2H'], n_trials=100, cv_folds=5)
model.optimize(timeout=3600)  # Limite de temps de 2 heures
model.get_best_model()
predictions = model.predict(X_valid_final)

encoded_columns = [col for col in X_valid_encoded.columns if col.startswith('ENTITY_DESCRIPTION_SHORT_')]
decoded_categories = X_valid_encoded[encoded_columns].idxmax(axis=1).apply(lambda x: x.replace('ENTITY_DESCRIPTION_SHORT_', ''))
X_valid_encoded['ENTITY_DESCRIPTION_SHORT'] = decoded_categories
X_valid_encoded['DATETIME'] = X_valid_encoded['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
print(X_valid_encoded.columns)
output = pd.DataFrame({'DATETIME': X_valid_encoded['DATETIME'],'ENTITY_DESCRIPTION_SHORT':X_valid_encoded['ENTITY_DESCRIPTION_SHORT'],'y_pred': predictions,'KEY':['Validation'for i in range(len(predictions))]})
output.to_csv('/Users/home/Documents/Hackathon/data/predictions_new.csv', index=False)
print ("Fini")