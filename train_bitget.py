import ccxt
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import time
from typing import Dict, Tuple
import json
import logging
from master import MASTERModel
import sys
import os
from pathlib import Path

class BitgetTrainer:
    def __init__(
        self,
        config_path: str = 'config/bitget_config.json',
        model_path: str = 'model/crypto_master_latest.pkl'
    ):
        # Configuration logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/training.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Chargement de la configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialisation de l'exchange
        self.exchange = ccxt.bitget({
            'apiKey': self.config['api_key'],
            'secret': self.config['api_secret'],
            'password': self.config['passphrase'],
            'enableRateLimit': True
        })

        # Paramètres du modèle
        self.model_params = {
            'd_feat': 40,
            'd_model': 256,
            't_nhead': 4,
            's_nhead': 2,
            'gate_input_start_index': 40,
            'gate_input_end_index': 60,
            'T_dropout_rate': 0.3,
            'S_dropout_rate': 0.3,
            'beta': 2.0,
            'n_epochs': 10,
            'lr': 1e-5,
            'GPU': 0 if torch.cuda.is_available() else None,
            'seed': 0,
            'train_stop_loss_thred': 0.95
        }

        # Initialisation du modèle
        self.model = self._init_or_load_model(model_path)
        
        # Paramètres d'entraînement
        self.batch_size = self.config.get('batch_size', 32)
        self.update_interval = self.config.get('update_interval', 3600)  # 1 heure
        self.min_samples = self.config.get('min_samples', 1000)

        # Buffer de données
        self.data_buffer = {
            'features': [],
            'labels': [],
            'timestamps': []
        }

    def _init_or_load_model(self, model_path: str) -> MASTERModel:
        """Initialise ou charge un modèle existant"""
        model = MASTERModel(**self.model_params)
        
        if os.path.exists(model_path):
            self.logger.info(f"Chargement du modèle depuis {model_path}")
            model.load_param(model_path)
        else:
            self.logger.info("Initialisation d'un nouveau modèle")
            
        return model

    def fetch_latest_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Récupère les dernières données de Bitget"""
        try:
            # Récupération OHLCV
            ohlcv = self.exchange.fetch_ohlcv(
                'DOGE/USDT',
                timeframe='1m',
                limit=self.config['lookback_window']
            )
            
            # Conversion en DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Calcul des features
            features = self._calculate_features(df)
            label = self._calculate_label(df)
            
            return features, label
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données: {e}")
            return None, None

    def _calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        """Calcule toutes les features techniques"""
        features = []
        
        # Prix normalisés
        for col in ['open', 'high', 'low', 'close']:
            series = df[col].values
            features.append(self._normalize_series(series))
        
        # Volumes
        vol = df['volume'].values
        features.append(self._normalize_series(vol))
        
        # Returns
        returns = np.diff(np.log(df['close'].values))
        features.append(np.pad(returns, (1,0), 'constant'))
        
        # Volatilité
        for window in [5, 15, 30]:
            vol = pd.Series(returns).rolling(window).std().values
            features.append(self._normalize_series(vol))
        
        # RSI
        for window in [6, 14, 24]:
            rsi = self._calculate_rsi(df['close'], window)
            features.append(rsi)
        
        # Moyennes mobiles
        for window in [7, 25, 99]:
            ma = df['close'].rolling(window).mean().values
            features.append(self._normalize_series(ma))
        
        # MACD
        macd, signal = self._calculate_macd(df['close'])
        features.extend([macd, signal])
        
        return np.stack(features, axis=1)

    def _normalize_series(self, series: np.ndarray) -> np.ndarray:
        """Normalisation robuste d'une série"""
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        normalized = (series - median) / (mad + 1e-8)
        return np.clip(normalized, -3, 3)

    def _calculate_rsi(self, prices: pd.Series, window: int) -> np.ndarray:
        """Calcule le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values

    def _calculate_macd(self, prices: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Calcule le MACD"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd.values, signal.values

    def train_iteration(self):
        """Effectue une itération d'entraînement"""
        try:
            # Récupération des données
            features, label = self.fetch_latest_data()
            if features is None or label is None:
                return
            
            # Ajout au buffer
            self.data_buffer['features'].append(features)
            self.data_buffer['labels'].append(label)
            self.data_buffer['timestamps'].append(datetime.now())
            
            # Nettoyage du buffer (garde les dernières 24h)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self._clean_buffer(cutoff_time)
            
            # Entraînement si assez de données
            if len(self.data_buffer['features']) >= self.min_samples:
                # Préparation des données
                train_features = np.stack(self.data_buffer['features'][-self.min_samples:])
                train_labels = np.array(self.data_buffer['labels'][-self.min_samples:])
                
                # Mise à jour du modèle
                self.model.update(train_features, train_labels)
                
                # Sauvegarde du modèle
                self.save_model()
                
                self.logger.info("Itération d'entraînement terminée")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement: {e}")

    def _clean_buffer(self, cutoff_time: datetime):
        """Nettoie le buffer des anciennes données"""
        while (len(self.data_buffer['timestamps']) > 0 and 
               self.data_buffer['timestamps'][0] < cutoff_time):
            for key in self.data_buffer:
                self.data_buffer[key].pop(0)

    def save_model(self):
        """Sauvegarde le modèle"""
        try:
            path = f"model/crypto_master_{int(time.time())}.pkl"
            torch.save(self.model.state_dict(), path)
            latest_path = "model/crypto_master_latest.pkl"
            torch.save(self.model.state_dict(), latest_path)
            self.logger.info(f"Modèle sauvegardé: {path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du modèle: {e}")

    def run(self):
        """Boucle principale d'entraînement"""
        self.logger.info("Démarrage de l'entraînement continu")
        
        while True:
            try:
                self.train_iteration()
                time.sleep(self.update_interval)
            except KeyboardInterrupt:
                self.logger.info("Arrêt de l'entraînement")
                break
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle principale: {e}")
                time.sleep(60)  # Attente avant nouvelle tentative

if __name__ == "__main__":
    # Création des dossiers nécessaires
    for folder in ['logs', 'model', 'config']:
        Path(folder).mkdir(exist_ok=True)
        
    # Configuration par défaut
    if not os.path.exists('config/bitget_config.json'):
        default_config = {
            'api_key': '',
            'api_secret': '',
            'passphrase': '',
            'batch_size': 32,
            'update_interval': 3600,
            'min_samples': 1000,
            'lookback_window': 120
        }
        with open('config/bitget_config.json', 'w') as f:
            json.dump(default_config, f, indent=4)
            
    trainer = BitgetTrainer()
    trainer.run()