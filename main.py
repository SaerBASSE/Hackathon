import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import logging
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import joblib

warnings.filterwarnings("ignore")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class XGBoostOptimizer:
    """
    Classe optimisée pour l'entraînement et l'optimisation d'XGBoost
    avec validation robuste et sauvegarde de modèle.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, n_trials: int = 100, 
                 cv_folds: int = 5, test_size: float = 0.2, random_state: int = 42):
        """
        Initialise l'optimiseur XGBoost.
        
        Args:
            X: Features d'entraînement
            y: Variable cible
            n_trials: Nombre d'essais pour l'optimisation
            cv_folds: Nombre de plis pour la validation croisée
            test_size: Proportion des données pour le test
            random_state: Graine aléatoire pour la reproductibilité
        """
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Division train/test pour validation finale
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.best_model = None
        self.study = None
        self.best_score = None
        
        logger.info(f"Données initialisées: Train({len(self.X_train)}), Test({len(self.X_test)})")
    
    def objective(self, trial) -> float:
        """
        Fonction objective pour l'optimisation Optuna.
        
        Args:
            trial: Essai Optuna
            
        Returns:
            Score RMSE (négatif pour minimisation)
        """
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'tree_method': 'hist',
            'n_jobs': -1,
            'random_state': self.random_state,
            'verbosity': 0,
            'objective': 'reg:squarederror'
        }
        
        model = xgb.XGBRegressor(**param)
        
        # Validation croisée sur les données d'entraînement uniquement
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=self.cv_folds,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        return -np.mean(scores)
    
    def optimize(self, timeout: int = 3600) -> Dict[str, Any]:
        """
        Lance l'optimisation des hyperparamètres.
        
        Args:
            timeout: Temps limite en secondes
            
        Returns:
            Meilleurs paramètres trouvés
        """
        logger.info("Début de l'optimisation hyperparamètres")
        
        self.study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20)
        )
        
        def callback(study, trial):
            if trial.number % 10 == 0:
                logger.info(f"Trial {trial.number}: Meilleur RMSE: {study.best_value:.6f}")
        
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=timeout,
            callbacks=[callback],
            show_progress_bar=True
        )
        
        self.best_score = self.study.best_value
        logger.info(f"Optimisation terminée. Meilleur RMSE: {self.best_score:.6f}")
        
        return self.study.best_params
    
    def train_best_model(self) -> None:
        """
        Entraîne le modèle final avec les meilleurs paramètres.
        """
        if self.study is None:
            raise ValueError("Vous devez d'abord exécuter optimize() avant d'entraîner le modèle.")
        
        best_params = self.study.best_params.copy()
        best_params.update({
            'tree_method': 'hist',
            'n_jobs': -1,
            'random_state': self.random_state,
            'verbosity': 0,
            'objective': 'reg:squarederror'
        })
        
        self.best_model = xgb.XGBRegressor(**best_params)
        
        # Entraînement sur toutes les données d'entraînement
        self.best_model.fit(self.X_train, self.y_train)
        
        # Évaluation sur le set de test
        test_predictions = self.best_model.predict(self.X_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_predictions))
        test_r2 = r2_score(self.y_test, test_predictions)
        
        logger.info(f"Performance sur le set de test - RMSE: {test_rmse:.6f}, R²: {test_r2:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Effectue des prédictions avec le meilleur modèle.
        
        Args:
            X: Données à prédire
            
        Returns:
            Prédictions
        """
        if self.best_model is None:
            raise ValueError("Vous devez d'abord entraîner le modèle avec train_best_model().")
        
        return self.best_model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Sauvegarde le modèle entraîné.
        
        Args:
            filepath: Chemin de sauvegarde
        """
        if self.best_model is None:
            raise ValueError("Aucun modèle entraîné à sauvegarder.")
        
        joblib.dump(self.best_model, filepath)
        logger.info(f"Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Charge un modèle sauvegardé.
        
        Args:
            filepath: Chemin du modèle
        """
        self.best_model = joblib.load(filepath)
        logger.info(f"Modèle chargé: {filepath}")


def load_and_preprocess_data(train_path: str, weather_path: str, 
                           test_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Charge et prétraite les données avec une gestion robuste des erreurs.
    
    Args:
        train_path: Chemin des données d'entraînement
        weather_path: Chemin des données météo
        test_path: Chemin des données de test
        
    Returns:
        X_scaled, y, X_test_scaled, metadata_test
    """
    logger.info("Chargement des données...")
    
    # Chargement des données
    try:
        df_train = pd.read_csv(train_path)
        df_weather = pd.read_csv(weather_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError as e:
        logger.error(f"Fichier non trouvé: {e}")
        raise
    
    # Nettoyage des données météo
    df_weather = df_weather.fillna(0)
    
    # Fusion des données d'entraînement
    df_train = pd.merge(df_train, df_weather, on='DATETIME', how='inner')
    
    # Extraction des features temporelles
    def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extrait les features temporelles d'une colonne DATETIME."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['DATETIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        # Vérification des dates invalides
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"{invalid_dates} dates invalides détectées et supprimées")
            df = df.dropna(subset=['date'])
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        
        return df
    
    df_train = extract_time_features(df_train)
    
    # Encodage des variables catégorielles
    df_train_encoded = pd.get_dummies(df_train, columns=['ENTITY_DESCRIPTION_SHORT'], drop_first=True)
    
    # Définition des features
    time_features = ['year', 'month', 'day', 'hour', 'minute', 'day_of_week', 'day_of_year']
    numerical_features = ['ADJUST_CAPACITY', 'CURRENT_WAIT_TIME', 'DOWNTIME', 
                         'feels_like', 'rain_1h', 'snow_1h','humidity','pressure','temp','wind_speed','clouds_all']
    categorical_features = [col for col in df_train_encoded.columns 
                          if col.startswith('ENTITY_DESCRIPTION_SHORT_')]
    
    all_features = time_features + numerical_features + categorical_features
    
    # Préparation des données d'entraînement
    X_train = df_train_encoded[all_features].copy()
    
    # Gestion des valeurs manquantes
    missing_cols = ['DOWNTIME', 'snow_1h', 'rain_1h']
    for col in missing_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna(0)
    
    y_train = df_train['WAIT_TIME_IN_2H'].values
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Préparation des données de test
    df_test = pd.merge(df_test, df_weather, on='DATETIME', how='inner')
    df_test = extract_time_features(df_test)
    df_test_encoded = pd.get_dummies(df_test, columns=['ENTITY_DESCRIPTION_SHORT'], drop_first=True)
    
    # Alignement des colonnes avec les données d'entraînement
    X_test = df_test_encoded.reindex(columns=X_train.columns, fill_value=0)
    
    # Gestion des valeurs manquantes pour le test
    for col in missing_cols:
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(0)
    
    X_test_scaled = scaler.transform(X_test)
    
    # Métadonnées pour l'export
    metadata_test = df_test[['DATETIME', 'ENTITY_DESCRIPTION_SHORT']].copy()
    
    logger.info(f"Données préparées - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    
    return X_train_scaled, y_train, X_test_scaled, metadata_test, scaler


def main(train =True):
    """Fonction principale d'exécution."""
    
    # Configuration des chemins (à adapter selon votre environnement)
    BASE_PATH = Path('/Users/home/Documents/Hackathon/data')
    TRAIN_PATH = BASE_PATH / 'waiting_times_train.csv'
    WEATHER_PATH = BASE_PATH / 'weather_data.csv'
    TEST_PATH = BASE_PATH / 'waiting_times_X_test_val.csv'
    OUTPUT_PATH = BASE_PATH / 'predictions_optimized.csv'
    MODEL_PATH = BASE_PATH / 'xgboost_model.joblib'
    
    try:
        # Chargement et prétraitement des données
        X_train, y_train, X_test, metadata_test, scaler = load_and_preprocess_data(
            str(TRAIN_PATH), str(WEATHER_PATH), str(TEST_PATH)
        )
        if train == True:
            # Optimisation du modèle
            optimizer = XGBoostOptimizer(
                X_train, y_train, 
                n_trials=50,  # Réduit pour les tests
                cv_folds=5
         )
        
            # Optimisation des hyperparamètres
            best_params = optimizer.optimize(timeout=3600)
            logger.info(f"Meilleurs paramètres: {best_params}")
        
            # Entraînement du modèle final
            optimizer.train_best_model()
        
            # Sauvegarde du modèle
            optimizer.save_model(str(MODEL_PATH))
        else:
            # Chargement du modèle existant
            optimizer = XGBoostOptimizer(X_train, y_train)
            optimizer.load_model(str(MODEL_PATH))
        # Prédictions
        predictions = optimizer.predict(X_test)
        
        # Export des résultats
        output_df = pd.DataFrame({
            'DATETIME': metadata_test['DATETIME'],
            'ENTITY_DESCRIPTION_SHORT': metadata_test['ENTITY_DESCRIPTION_SHORT'],
            'y_pred': predictions,
            'KEY': ['Validation'] * len(predictions)
        })
        
        output_df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Prédictions sauvegardées: {OUTPUT_PATH}")
        
        # Statistiques finales
        logger.info(f"Prédictions - Min: {predictions.min():.2f}, "
                   f"Max: {predictions.max():.2f}, "
                   f"Moyenne: {predictions.mean():.2f}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
        raise


if __name__ == "__main__":
    main(train=False)