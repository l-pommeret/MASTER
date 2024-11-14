import ccxt
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import time
import json
import logging
from pathlib import Path
from master import MASTERModel
import threading
from typing import Dict, Tuple

class BitgetTrader:
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
                logging.FileHandler('logs/trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Chargement configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialisation exchange
        self.exchange = ccxt.bitget({
            'apiKey': self.config['api_key'],
            'secret': self.config['api_secret'],
            'password': self.config['passphrase'],
            'enableRateLimit': True
        })

        # Paramètres de trading
        self.symbol = 'DOGE/USDT'
        self.position = None
        self.position_size = self.config.get('position_size', 100)  # USDT
        self.stop_loss = self.config.get('stop_loss', 0.02)        # 2%
        self.take_profit = self.config.get('take_profit', 0.04)    # 4%
        self.min_confidence = self.config.get('min_confidence', 0.7) # Seuil de confiance

        # Initialisation du modèle
        self.model = self._load_model(model_path)
        self.model_lock = threading.Lock()
        
        # État du trader
        self.last_prediction = None
        self.entry_price = None
        self.last_update = None
        
    def _load_model(self, model_path: str) -> MASTERModel:
        """Charge le modèle avec les paramètres par défaut"""
        model_params = {
            'd_feat': 40,
            'd_model': 256,
            't_nhead': 4,
            's_nhead': 2,
            'gate_input_start_index': 40,
            'gate_input_end_index': 60,
            'T_dropout_rate': 0.3,
            'S_dropout_rate': 0.3,
            'beta': 2.0,
        }
        
        model = MASTERModel(**model_params)
        model.load_param(model_path)
        return model

    async def _check_model_update(self):
        """Vérifie et charge les mises à jour du modèle"""
        while True:
            try:
                model_path = 'model/crypto_master_latest.pkl'
                if os.path.exists(model_path):
                    mod_time = os.path.getmtime(model_path)
                    if self.last_update is None or mod_time > self.last_update:
                        with self.model_lock:
                            self.logger.info("Mise à jour du modèle détectée")
                            self.model.load_param(model_path)
                            self.last_update = mod_time
            except Exception as e:
                self.logger.error(f"Erreur lors de la vérification du modèle: {e}")
            
            await asyncio.sleep(60)  # Vérifie toutes les minutes

    def get_current_position(self) -> Dict:
        """Récupère la position actuelle"""
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    return pos
            return None
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de la position: {e}")
            return None

    def place_order(self, side: str, amount: float, price: float = None) -> bool:
        """Place un ordre sur Bitget"""
        try:
            order_params = {
                'symbol': self.symbol,
                'type': 'market' if price is None else 'limit',
                'side': side,
                'amount': amount
            }
            if price is not None:
                order_params['price'] = price
                
            order = self.exchange.create_order(**order_params)
            self.logger.info(f"Ordre placé: {order}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du placement de l'ordre: {e}")
            return False

    def manage_position(self, prediction: float, confidence: float):
        """Gère la position en fonction de la prédiction"""
        try:
            current_position = self.get_current_position()
            current_price = self.exchange.fetch_ticker(self.symbol)['last']

            # Si pas de position
            if current_position is None or float(current_position['size']) == 0:
                if abs(confidence) >= self.min_confidence:
                    amount = self.position_size / current_price
                    if prediction > 0:
                        if self.place_order('buy', amount):
                            self.entry_price = current_price
                    else:
                        if self.place_order('sell', amount):
                            self.entry_price = current_price

            # Si position existante
            else:
                position_size = float(current_position['size'])
                position_side = current_position['side']
                
                # Vérification stop loss / take profit
                if position_side == 'long':
                    profit_pct = (current_price - self.entry_price) / self.entry_price
                    if profit_pct <= -self.stop_loss or profit_pct >= self.take_profit:
                        self.place_order('sell', position_size)
                else:
                    profit_pct = (self.entry_price - current_price) / self.entry_price
                    if profit_pct <= -self.stop_loss or profit_pct >= self.take_profit:
                        self.place_order('buy', position_size)
                        
        except Exception as e:
            self.logger.error(f"Erreur dans la gestion de position: {e}")

    async def run_trading(self):
        """Boucle principale de trading"""
        self.logger.info("Démarrage du trading bot")
        
        # Démarrage de la vérification des mises à jour du modèle
        model_checker = asyncio.create_task(self._check_model_update())
        
        while True:
            try:
                # Récupération des données
                features = self.prepare_features()
                
                # Prédiction
                with self.model_lock:
                    prediction = self.model.predict(features)
                confidence = self.calculate_confidence(prediction)
                
                self.logger.info(f"Prédiction: {prediction:.4f}, Confiance: {confidence:.4f}")
                
                # Gestion de la position
                self.manage_position(prediction, confidence)
                
                # Attente avant prochain cycle
                await asyncio.sleep(60)  # Vérifie toutes les minutes
                
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle de trading: {e}")
                await asyncio.sleep(60)

    def calculate_confidence(self, prediction: float) -> float:
        """Calcule le niveau de confiance de la prédiction"""
        return abs(prediction)  # Simplifié pour l'exemple

if __name__ == "__main__":
    # Création des dossiers nécessaires
    for folder in ['logs', 'config']:
        Path(folder).mkdir(exist_ok=True)
        
    # Vérification de la configuration
    if not os.path.exists('config/bitget_config.json'):
        print("Veuillez créer le fichier config/bitget_config.json")
        exit(1)
        
    trader = BitgetTrader()
    
    # Lancement du trading avec asyncio
    import asyncio
    asyncio.run(trader.run_trading())