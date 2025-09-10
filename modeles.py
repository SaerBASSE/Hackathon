import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import os
import multiprocessing

# Optimiser pour M4 Pro
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())

class OptimizedEnsembleRegressor:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        
    def _init_models(self, n_features):
        """Initialise les modèles optimisés pour M4"""
        self.models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                n_jobs=-1,
                random_state=42
            ),
            
            'lgb': lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.1,
                num_leaves=31,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                num_threads=-1,
                random_state=42,
                verbose=-1
            ),
            
            'catboost': CatBoostRegressor(
                iterations=1000,
                learning_rate=0.1,
                depth=6,
                thread_count=-1,
                verbose=False,
                random_state=42
            ),
            
            'rf': RandomForestRegressor(
                n_estimators=600,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            ),
            
            'gb': GradientBoostingRegressor(
                n_estimators=600,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=42
            ),
            
            'ridge': Ridge(alpha=1.0),
            
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
        }
    
    def fit(self, X, y):
        """Entraîne l'ensemble de modèles"""
        
        X_scaled = self.scaler.fit_transform(X)
        self._init_models(X.shape[1])
        
        
        cv_scores = {}
        
        for name, model in self.models.items():
            print(f"Entraînement {name}...")
            
            
            if name in ['ridge', 'elastic', 'svr']:
                X_input = X_scaled
            else:
                X_input = X
            
            
            scores = cross_val_score(
                model, X_input, y, 
                cv=5, 
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )
            cv_scores[name] = -scores.mean()
            
            
            model.fit(X_input, y)
            
            print(f"   RMSE CV: {cv_scores[name]:.4f}")
        
        
        total_inverse_error = sum(1/score for score in cv_scores.values())
        self.weights = {
            name: (1/score) / total_inverse_error 
            for name, score in cv_scores.items()
        }
        
        print("\n Poids des modèles:")
        for name, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            print(f"   {name}: {weight:.4f}")
        
        return self
    
    def predict(self, X):
        """Prédiction par ensemble pondéré"""
        X_scaled = self.scaler.transform(X)
        predictions = []
        
        for name, model in self.models.items():
            if name in ['ridge', 'elastic', 'svr']:
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)
            predictions.append(pred * self.weights[name])
        
        return np.sum(predictions, axis=0)
    
    def get_feature_importance(self, feature_names=None):
        """Calcule l'importance des features agrégée"""
        importances = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                weight = self.weights[name]
                if name not in importances:
                    importances[name] = model.feature_importances_ * weight
        
        
        if importances:
            avg_importance = np.mean(list(importances.values()), axis=0)
            
            if feature_names is not None:
                return pd.Series(avg_importance, index=feature_names).sort_values(ascending=False)
            else:
                return avg_importance
        
        return None
