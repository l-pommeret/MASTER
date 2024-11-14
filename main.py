import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from master import MASTERModel
from typing import List, Tuple, Dict
import pickle
import os

class BitgetDataPreparator:
    def __init__(
        self,
        symbols: List[str] = ['DOGE/USDT', 'BTC/USDT', 'ETH/USDT'],
        timeframe: str = '1m',
        lookback_window: int = 120,  # 2 heures
        feature_window: int = 60,    # 1 heure pour certains calculs
        market_features: int = 20
    ):
        self.exchange = ccxt.bitget()
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback = lookback_window
        self.feature_window = feature_window
        self.market_features = market_features

    def fetch_ohlcv(self, symbol: str, start_time: int) -> pd.DataFrame:
        """Récupère les données OHLCV de Bitget"""
        try:
            data = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=self.timeframe,
                since=start_time,
                limit=self.lookback + self.feature_window
            )
            df = pd.DataFrame(
                data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.set_index('timestamp')
        except Exception as e:
            print(f"Erreur lors de la récupération des données pour {symbol}: {e}")
            return None

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les indicateurs techniques"""
        features = pd.DataFrame(index=df.index)
        
        # Rendements
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close']).diff()
        
        # Volatilité
        returns = features['returns']
        for window in [5, 15, 30, 60]:
            features[f'volatility_{window}m'] = returns.rolling(window).std()
        
        # RSI
        for window in [6, 14, 24]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            features[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # Moyennes mobiles
        for window in [7, 25, 99]:
            features[f'sma_{window}'] = df['close'].rolling(window).mean()
            features[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Volumes
        features['volume_sma_10'] = df['volume'].rolling(10).mean()
        features['volume_ema_10'] = df['volume'].ewm(span=10).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_10']
        
        return features

    def calculate_market_features(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calcule les features de marché"""
        market_features = pd.DataFrame(index=all_data['DOGE/USDT'].index)
        
        # BTC comme proxy du marché
        btc_data = all_data['BTC/USDT']
        btc_returns = btc_data['close'].pct_change()
        
        # Features de marché
        for window in [5, 15, 30, 60]:
            market_features[f'market_volatility_{window}m'] = btc_returns.rolling(window).std()
            market_features[f'market_trend_{window}m'] = btc_returns.rolling(window).mean()
            
        # Corrélations
        doge_returns = all_data['DOGE/USDT']['close'].pct_change()
        for window in [10, 30, 60]:
            market_features[f'btc_correlation_{window}m'] = (
                doge_returns.rolling(window)
                .corr(btc_returns)
            )
            
        # Volume de marché
        market_volume = sum(df['volume'].fillna(0) for df in all_data.values())
        market_features['market_volume'] = market_volume
        market_features['market_volume_ma'] = market_volume.rolling(30).mean()
        
        return market_features

    def prepare_data(self, start_time: int = None) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """Prépare les données pour le modèle"""
        # Récupération des données
        all_data = {}
        for symbol in self.symbols:
            df = self.fetch_ohlcv(symbol, start_time)
            if df is not None:
                all_data[symbol] = df
        
        if not all_data:
            raise ValueError("Aucune donnée n'a pu être récupérée")
            
        # Calcul des features
        doge_features = self.calculate_technical_features(all_data['DOGE/USDT'])
        market_features = self.calculate_market_features(all_data)
        
        # Concaténation et normalisation
        combined = pd.concat([doge_features, market_features], axis=1)
        combined = combined.fillna(method='ffill').fillna(0)
        
        # Normalisation robuste
        for column in combined.columns:
            median = combined[column].median()
            mad = (combined[column] - median).abs().median()
            combined[column] = (combined[column] - median) / (mad + 1e-8)
            combined[column] = combined[column].clip(-3, 3)
        
        # Création des séquences
        sequences = []
        timestamps = []
        
        for i in range(len(combined) - self.lookback):
            sequences.append(combined.iloc[i:i+self.lookback].values)
            timestamps.append(combined.index[i+self.lookback])
            
        return np.array(sequences), pd.DatetimeIndex(timestamps)

def main():
    # Paramètres
    d_feat = 40            # Nombre de features techniques
    d_model = 256         # Dimension du modèle
    t_nhead = 4          # Têtes d'attention temporelle
    s_nhead = 2          # Têtes d'attention spatiale
    dropout = 0.3        # Taux de dropout
    gate_input_start_index = 40   # Début des features marché
    gate_input_end_index = 60    # Fin des features marché
    beta = 2.0           # Paramètre de température du gate
    
    # Paramètres d'entraînement
    n_epoch = 20
    lr = 1e-5
    GPU = 0 if torch.cuda.is_available() else None
    seed = 0
    train_stop_loss_thred = 0.95

    # Préparation des données
    data_prep = BitgetDataPreparator()
    
    # Division train/valid/test
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = end_time - (86400 * 1000 * 30)  # 30 jours de données
    
    sequences, timestamps = data_prep.prepare_data(start_time)
    
    # 70% train, 15% validation, 15% test
    train_idx = int(len(sequences) * 0.7)
    valid_idx = int(len(sequences) * 0.85)
    
    train_data = sequences[:train_idx]
    valid_data = sequences[train_idx:valid_idx]
    test_data = sequences[valid_idx:]
    
    # Sauvegarde des données préparées
    os.makedirs('data/crypto', exist_ok=True)
    with open('data/crypto/dl_train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('data/crypto/dl_valid.pkl', 'wb') as f:
        pickle.dump(valid_data, f)
    with open('data/crypto/dl_test.pkl', 'wb') as f:
        pickle.dump(test_data, f)

    # Initialisation du modèle
    model = MASTERModel(
        d_feat=d_feat,
        d_model=d_model,
        t_nhead=t_nhead,
        s_nhead=s_nhead,
        gate_input_start_index=gate_input_start_index,
        gate_input_end_index=gate_input_end_index,
        T_dropout_rate=dropout,
        S_dropout_rate=dropout,
        beta=beta,
        n_epochs=n_epoch,
        lr=lr,
        GPU=GPU,
        seed=seed,
        train_stop_loss_thred=train_stop_loss_thred,
        save_path='model/',
        save_prefix='crypto'
    )

    # Entraînement
    print("Début de l'entraînement...")
    model.fit(train_data, valid_data)
    print("Entraînement terminé.")

    # Test
    predictions, metrics = model.predict(test_data)
    print("Métriques sur le jeu de test:")
    print(metrics)

    # Sauvegarde des prédictions
    predictions.to_csv('predictions.csv')

if __name__ == "__main__":
    main()