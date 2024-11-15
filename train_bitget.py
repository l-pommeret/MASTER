import ccxt
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import time
from typing import Dict, Tuple, Optional
import json
import logging
import sys
import os
from pathlib import Path
from master import MASTERModel

class BitgetTrainer:
    def __init__(
        self,
        config_path: str = 'config/bitget_config.json',
        model_path: str = 'model/crypto_master_latest.pkl',
        log_level: str = 'INFO'
    ):
        # Configuration des logs
        self._setup_logging(log_level)
        self.logger.info("Initialisation du BitgetTrainer...")

        # Chargement de la configuration
        self.config = self._load_config(config_path)
        
        # Métriques de suivi
        self.training_metrics = {
            'iterations': 0,
            'total_samples': 0,
            'best_loss': float('inf'),
            'best_accuracy': 0.0,
            'last_save': None,
            'training_start': datetime.now().isoformat()
        }

        # Initialisation de l'exchange
        self.exchange = self._init_exchange()
        
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
        self.update_interval = self.config.get('update_interval', 60)  # 1 minute
        self.min_samples = self.config.get('min_samples', 1000)
        self.lookback_window = self.config.get('lookback_window', 120)

        # Buffer de données
        self.data_buffer = {
            'features': [],
            'labels': [],
            'timestamps': []
        }

        self.logger.info("Initialisation terminée")
        self._log_config()

    def _setup_logging(self, log_level: str):
        """Configure le système de logging"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Création d'un fichier de log daté
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configuration du logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> Dict:
        """Charge la configuration depuis le fichier json"""
        try:
            self.logger.info(f"Chargement de la configuration depuis {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {e}")
            raise

    def _init_exchange(self) -> ccxt.Exchange:
        """Initialise la connexion à l'exchange"""
        try:
            self.logger.info("Initialisation de la connexion à Bitget")
            return ccxt.bitget({
                'apiKey': self.config['api_key'],
                'secret': self.config['api_secret'],
                'password': self.config['passphrase'],
                'enableRateLimit': True
            })
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation de l'exchange: {e}")
            raise

    def _log_config(self):
        """Log la configuration actuelle"""
        self.logger.info("Configuration actuelle:")
        self.logger.info(f"- Batch size: {self.batch_size}")
        self.logger.info(f"- Update interval: {self.update_interval}s")
        self.logger.info(f"- Min samples: {self.min_samples}")
        self.logger.info(f"- Lookback window: {self.lookback_window}")
        if torch.cuda.is_available():
            self.logger.info(f"- GPU disponible: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("- Mode CPU")

    def _init_or_load_model(self, model_path: str) -> MASTERModel:
        """Initialise ou charge un modèle existant"""
        try:
            model = MASTERModel(**self.model_params)
            
            if os.path.exists(model_path):
                self.logger.info(f"Chargement du modèle depuis {model_path}")
                model.load_param(model_path)
                self.logger.info("Modèle chargé avec succès")
            else:
                self.logger.info("Initialisation d'un nouveau modèle")
                
            return model
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation du modèle: {e}")
            raise

    def fetch_latest_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Récupère les dernières données de Bitget avec vérification"""
        try:
            self.logger.debug("Récupération des données OHLCV...")
            
            # Récupération OHLCV
            ohlcv = self.exchange.fetch_ohlcv(
                'DOGE/USDT',
                timeframe='1m',
                limit=self.lookback_window
            )
            
            # Vérification de la taille des données
            if len(ohlcv) < self.lookback_window:
                self.logger.warning(f"Données incomplètes: {len(ohlcv)} < {self.lookback_window}")
                return None, None
                
            # Conversion en DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Vérification des valeurs manquantes
            if df.isna().any().any():
                self.logger.warning("Valeurs manquantes détectées - Application du forward/backward fill")
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Calcul des features
            features = self._calculate_features(df)
            
            # Calcul du label (retour futur)
            current_price = df['close'].iloc[-1]
            previous_price = df['close'].iloc[-2]
            label = (current_price / previous_price) - 1
            
            self.logger.debug(f"Données récupérées - Features shape: {features.shape}")
            return features, np.array([label])
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données: {e}", exc_info=True)
            return None, None

    def _calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        """Calcule toutes les features techniques"""
        features = []
        length = len(df)
        
        try:
            # Prix normalisés
            for col in ['open', 'high', 'low', 'close']:
                series = df[col].values
                normalized = self._normalize_series(series)
                features.append(normalized)
            
            # Volumes
            vol = df['volume'].values
            normalized_vol = self._normalize_series(vol)
            features.append(normalized_vol)
            
            # Returns
            returns = np.diff(np.log(df['close'].values))
            padded_returns = np.pad(returns, (1,0), 'constant')
            features.append(padded_returns)
            
            # Volatilité
            returns_series = pd.Series(padded_returns)
            for window in [5, 15, 30]:
                vol = returns_series.rolling(window).std().fillna(0).values
                normalized_vol = self._normalize_series(vol)
                features.append(normalized_vol)
            
            # RSI
            for window in [6, 14, 24]:
                rsi = self._calculate_rsi(df['close'], window)
                features.append(rsi)
            
            # Moyennes mobiles
            for window in [7, 25, 99]:
                ma = df['close'].rolling(window).mean().fillna(0).values
                normalized_ma = self._normalize_series(ma)
                features.append(normalized_ma)
            
            # MACD
            macd, signal = self._calculate_macd(df['close'])
            features.extend([macd, signal])
            
            # Vérification finale des dimensions
            all_features = np.stack(features, axis=1)
            self.logger.debug(f"Features calculées - Shape: {all_features.shape}")
            
            return all_features
            
        except Exception as e:
            self.logger.error(f"Erreur dans le calcul des features: {str(e)}")
            self.logger.error(f"Longueur du DataFrame: {length}")
            for i, feature in enumerate(features):
                self.logger.error(f"Feature {i} shape: {feature.shape}")
            raise

    def _normalize_series(self, series: np.ndarray) -> np.ndarray:
        """Normalisation robuste d'une série"""
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        normalized = (series - median) / (mad + 1e-8)
        return np.clip(normalized, -3, 3)

    def _calculate_rsi(self, prices: pd.Series, window: int) -> np.ndarray:
        """Calcule le RSI avec gestion des NaN"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values

    def _calculate_macd(self, prices: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Calcule le MACD avec gestion des NaN"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        macd = macd.fillna(0).values
        signal = signal.fillna(0).values
        return macd, signal

    def calculate_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        """Calcule les métriques de performance"""
        mse_loss = np.mean((predictions - labels) ** 2)
        direction_accuracy = np.mean(np.sign(predictions) == np.sign(labels))
        
        return {
            'mse_loss': float(mse_loss),
            'direction_accuracy': float(direction_accuracy),
            'timestamp': datetime.now().isoformat(),
            'iteration': self.training_metrics['iterations']
        }

    def should_save_model(self, current_loss: float, current_accuracy: float) -> bool:
        """Détermine si le modèle doit être sauvegardé"""
        should_save = False
        reasons = []
        
        # Sauvegarde si meilleure loss
        if current_loss < self.training_metrics['best_loss']:
            self.training_metrics['best_loss'] = current_loss
            should_save = True
            reasons.append(f"Nouvelle meilleure loss: {current_loss:.6f}")
            
        # Sauvegarde si meilleure accuracy
        if current_accuracy > self.training_metrics['best_accuracy']:
            self.training_metrics['best_accuracy'] = current_accuracy
            should_save = True
            reasons.append(f"Nouvelle meilleure accuracy: {current_accuracy:.4f}")
            
        # Sauvegarde périodique (toutes les heures)
        if (self.training_metrics['last_save'] is None or 
            time.time() - self.training_metrics['last_save'] > 3600):
            should_save = True
            reasons.append("Sauvegarde périodique")
            
        if reasons:
            self.logger.info("Raisons de la sauvegarde: " + ", ".join(reasons))
            
        return should_save

    def save_model(self, metrics: Optional[Dict] = None):
        """Sauvegarde le modèle et ses métriques"""
        try:
            timestamp = int(time.time())
            
            # Sauvegarde du modèle
            model_path = f"model/crypto_master_{timestamp}.pkl"
            torch.save(self.model.state_dict(), model_path)
            
            # Sauvegarde comme latest
            latest_path = "model/crypto_master_latest.pkl"
            torch.save(self.model.state_dict(), latest_path)
            
            # Sauvegarde des métriques
            if metrics:
                # Ajout des métriques globales
                metrics.update({
                    'total_iterations': self.training_metrics['iterations'],
                    'total_samples': self.training_metrics['total_samples'],
                    'best_loss': self.training_metrics['best_loss'],
                    'best_accuracy': self.training_metrics['best_accuracy'],
                    'training_start': self.training_metrics['training_start']
                })
                
                metrics_path = f"model/metrics_{timestamp}.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
            
            self.training_metrics['last_save'] = timestamp
            self.logger.info(f"Modèle sauvegardé: {model_path}")
            if metrics:
                self.logger.info(f"Métriques sauvegardées: {metrics_path}")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}", exc_info=True)

    def train_iteration(self):
        """Effectue une itération d'entraînement avec gestion robuste des données"""
        iteration_start = time.time()
        
        try:
            self.logger.info(f"=== Début itération {self.training_metrics['iterations'] + 1} ===")
            
            # Récupération des données
            self.logger.info("Récupération des dernières données...")
            features, label = self.fetch_latest_data()
            
            if features is None or label is None:
                self.logger.warning("Pas de données disponibles pour cette itération")
                return
            
            # Vérification des dimensions
            self.logger.debug(f"Dimensions des features: {features.shape}")
            self.logger.debug(f"Dimensions du label: {label.shape}")
            
            # Ajout au buffer avec vérification
            try:
                self.data_buffer['features'].append(features)
                self.data_buffer['labels'].append(label)
                self.data_buffer['timestamps'].append(datetime.now())
                
                buffer_size = len(self.data_buffer['features'])
                self.logger.info(f"Buffer actuel: {buffer_size}/{self.min_samples} échantillons")
                
            except Exception as e:
                self.logger.error(f"Erreur lors de l'ajout au buffer: {e}")
                raise
            
            # Nettoyage du buffer des données anciennes
            try:
                cutoff_time = datetime.now() - timedelta(hours=24)
                before_clean = len(self.data_buffer['features'])
                self._clean_buffer(cutoff_time)
                after_clean = len(self.data_buffer['features'])
                
                if before_clean != after_clean:
                    self.logger.info(f"Nettoyage buffer: {before_clean} -> {after_clean} échantillons")
                    
            except Exception as e:
                self.logger.error(f"Erreur lors du nettoyage du buffer: {e}")
                raise
            
            # Entraînement si assez de données
            if len(self.data_buffer['features']) >= self.min_samples:
                self.logger.info("Préparation des données d'entraînement...")
                
                try:
                    # Préparation des données
                    features_array = np.stack(self.data_buffer['features'][-self.min_samples:])
                    labels_array = np.array(self.data_buffer['labels'][-self.min_samples:])
                    
                    # Vérification et log des dimensions
                    self.logger.debug(f"Shape des features: {features_array.shape}")
                    self.logger.debug(f"Shape des labels: {labels_array.shape}")
                    
                    # Vérification des NaN
                    if np.isnan(features_array).any() or np.isnan(labels_array).any():
                        self.logger.warning("Détection de NaN dans les données")
                        features_array = np.nan_to_num(features_array, nan=0.0)
                        labels_array = np.nan_to_num(labels_array, nan=0.0)
                    
                    # Division train/validation (80/20)
                    split_idx = int(len(features_array) * 0.8)
                    train_features = features_array[:split_idx]
                    train_labels = labels_array[:split_idx]
                    valid_features = features_array[split_idx:]
                    valid_labels = labels_array[split_idx:]
                    
                    # Log des dimensions après split
                    self.logger.debug(f"Train features shape: {train_features.shape}")
                    self.logger.debug(f"Train labels shape: {train_labels.shape}")
                    self.logger.debug(f"Valid features shape: {valid_features.shape}")
                    self.logger.debug(f"Valid labels shape: {valid_labels.shape}")
                    
                    # Entraînement du modèle
                    self.logger.info("Entraînement du modèle...")
                    
                    # Conversion en tenseurs PyTorch
                    train_features = torch.FloatTensor(train_features).to(self.device)
                    train_labels = torch.FloatTensor(train_labels).to(self.device)
                    valid_features = torch.FloatTensor(valid_features).to(self.device)
                    valid_labels = torch.FloatTensor(valid_labels).to(self.device)
                    
                    # Entraînement
                    try:
                        train_loss = self.model.fit(train_features, train_labels)
                        self.logger.info(f"Loss d'entraînement: {train_loss:.6f}")
                        
                        # Évaluation
                        with torch.no_grad():
                            predictions = self.model.predict(valid_features)
                            metrics = self.calculate_metrics(predictions.cpu().numpy(), valid_labels.cpu().numpy())
                        
                        self.logger.info(f"Résultats de l'itération {self.training_metrics['iterations']+1}:")
                        self.logger.info(f"- MSE Loss: {metrics['mse_loss']:.6f}")
                        self.logger.info(f"- Direction Accuracy: {metrics['direction_accuracy']:.4f}")
                        
                        # Vérification pour la sauvegarde
                        if self.should_save_model(metrics['mse_loss'], metrics['direction_accuracy']):
                            metrics['training_duration'] = time.time() - iteration_start
                            self.save_model(metrics)
                        
                        # Mise à jour des métriques
                        self.training_metrics['iterations'] += 1
                        self.training_metrics['total_samples'] += len(train_labels)
                        
                    except Exception as e:
                        self.logger.error(f"Erreur pendant l'entraînement: {str(e)}")
                        self.logger.error(f"Shapes - Train Features: {train_features.shape}, Train Labels: {train_labels.shape}")
                        raise
                    
                except Exception as e:
                    self.logger.error(f"Erreur lors de la préparation des données: {str(e)}")
                    raise
                
                iteration_duration = time.time() - iteration_start
                self.logger.info(f"Itération terminée en {iteration_duration:.2f} secondes")
                
                # Log des performances globales
                self.logger.info("=== Métriques globales ===")
                self.logger.info(f"Meilleures performances:")
                self.logger.info(f"- Best Loss: {self.training_metrics['best_loss']:.6f}")
                self.logger.info(f"- Best Accuracy: {self.training_metrics['best_accuracy']:.4f}")
                self.logger.info(f"Total échantillons traités: {self.training_metrics['total_samples']}")
                
            else:
                self.logger.info(f"Pas assez d'échantillons pour l'entraînement ({len(self.data_buffer['features'])}/{self.min_samples})")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement: {e}")
            self.logger.error("Stack trace complète:", exc_info=True)
            # Pas de raise ici pour permettre à la boucle principale de continuer

    def _clean_buffer(self, cutoff_time: datetime):
        """Nettoie le buffer des anciennes données"""
        try:
            removed = 0
            while (len(self.data_buffer['timestamps']) > 0 and 
                   self.data_buffer['timestamps'][0] < cutoff_time):
                for key in self.data_buffer:
                    self.data_buffer[key].pop(0)
                removed += 1
                
            if removed > 0:
                self.logger.debug(f"Nettoyage buffer: {removed} échantillons supprimés")
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage du buffer: {e}")

    def run(self):
        """Boucle principale d'entraînement"""
        self.logger.info("=== Démarrage de l'entraînement continu ===")
        self.logger.info(f"Intervalle de mise à jour: {self.update_interval} secondes")
        start_time = datetime.now()
        
        try:
            while True:
                try:
                    # Log du temps total d'exécution
                    runtime = datetime.now() - start_time
                    self.logger.info(f"Temps d'exécution total: {runtime}")
                    
                    # Exécution de l'itération
                    self.train_iteration()
                    
                    # Attente avant la prochaine itération
                    self.logger.info(f"Attente de {self.update_interval} secondes...")
                    time.sleep(self.update_interval)
                    
                except KeyboardInterrupt:
                    self.logger.info("Interruption manuelle détectée")
                    raise
                except Exception as e:
                    self.logger.error(f"Erreur dans la boucle d'entraînement: {e}")
                    self.logger.warning("Attente de 60 secondes avant nouvelle tentative...")
                    time.sleep(60)
                    
        except KeyboardInterrupt:
            self.logger.info("=== Arrêt de l'entraînement ===")
            self.logger.info(f"Durée totale de l'entraînement: {datetime.now() - start_time}")
            self.logger.info(f"Nombre total d'itérations: {self.training_metrics['iterations']}")
            self.logger.info(f"Nombre total d'échantillons: {self.training_metrics['total_samples']}")
            
            # Sauvegarde finale du modèle
            self.save_model({
                'final_metrics': True,
                'total_runtime': str(datetime.now() - start_time),
                'total_iterations': self.training_metrics['iterations'],
                'total_samples': self.training_metrics['total_samples'],
                'best_loss': self.training_metrics['best_loss'],
                'best_accuracy': self.training_metrics['best_accuracy']
            })

if __name__ == "__main__":
    # Création des dossiers nécessaires
    for folder in ['logs', 'model', 'config']:
        Path(folder).mkdir(exist_ok=True)
        
    # Configuration par défaut si nécessaire
    if not os.path.exists('config/bitget_config.json'):
        default_config = {
                'api_key': '',
                'api_secret': '',
                'passphrase': '',
                'batch_size': 32,
                'update_interval': 5,      # 5 secondes au lieu de 60
                'min_samples': 100,        # 100 échantillons au lieu de 1000
                'lookback_window': 60      # 60 minutes au lieu de 120
            }
        with open('config/bitget_config.json', 'w') as f:
            json.dump(default_config, f, indent=4)
            
    trainer = BitgetTrainer()
    trainer.run()