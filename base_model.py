import numpy as np
import pandas as pd
import copy
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim
import ccxt

def calc_ic(pred, label):
    df = pd.DataFrame({'pred': pred, 'label': label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

class CryptoMinuteBatchSampler(Sampler):
    def __init__(self, data_source, shuffle=False, batch_minutes=120):
        self.data_source = data_source
        self.shuffle = shuffle
        self.batch_minutes = batch_minutes
        
        # Calcul des indices de batch basés sur les timestamps
        self.minute_count = pd.Series(index=self.data_source.get_index()).groupby("timestamp").size().values
        self.minute_index = np.roll(np.cumsum(self.minute_count), 1)
        self.minute_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.minute_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.minute_index[i], self.minute_index[i] + self.minute_count[i])
        else:
            for idx, count in zip(self.minute_index, self.minute_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)

class CryptoSequenceModel():
    def __init__(self, 
                 n_epochs, 
                 lr, 
                 exchange_id='bitget',
                 symbol='DOGE/USDT',
                 GPU=None, 
                 seed=None, 
                 train_stop_loss_thred=None,
                 save_path='model/',
                 save_prefix=''):
        
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() and GPU is not None else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        
        # Paramètres crypto spécifiques
        self.exchange = getattr(ccxt, exchange_id)()
        self.symbol = symbol
        
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            
        self.fitted = False
        self.model = None
        self.train_optimizer = None
        
        self.save_path = save_path
        self.save_prefix = save_prefix

    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")
        
        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        # MSE modifié avec pénalité sur les faux signaux
        mask = ~torch.isnan(label)
        base_loss = (pred[mask] - label[mask])**2
        
        # Pénalité supplémentaire pour les mauvais sens
        direction_penalty = torch.where(
            pred[mask] * label[mask] < 0,
            torch.ones_like(pred[mask]) * 0.2,
            torch.zeros_like(pred[mask])
        )
        
        return torch.mean(base_loss + direction_penalty)

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []
        
        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, :-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            
            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())
            
            self.train_optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients pour stabilité
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
            self.train_optimizer.step()
            
        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for data in data_loader:
                data = torch.squeeze(data, dim=0)
                feature = data[:, :, :-1].to(self.device)
                label = data[:, -1, -1].to(self.device)
                
                pred = self.model(feature.float())
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())
                
        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = CryptoMinuteBatchSampler(data, shuffle=shuffle)
        return DataLoader(data, sampler=sampler, drop_last=drop_last)

    def load_param(self, param_path):
        self.model.load_state_dict(
            torch.load(param_path, map_location=self.device)
        )
        self.fitted = True

    def fit(self, train_features, train_labels):
        """Entraîne le modèle avec des données numpy"""
        self.model.train()
        
        # Conversion en tenseurs PyTorch
        features = torch.FloatTensor(train_features).to(self.device)
        labels = torch.FloatTensor(train_labels).to(self.device)
        
        best_loss = float('inf')
        best_param = None
        
        for epoch in range(self.n_epochs):
            self.train_optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(features)
            loss = self.loss_fn(predictions, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
            self.train_optimizer.step()
            
            current_loss = loss.item()
            print(f"Epoch {epoch}, loss: {current_loss:.6f}")
            
            # Sauvegarde du meilleur modèle
            if current_loss < best_loss:
                best_loss = current_loss
                best_param = copy.deepcopy(self.model.state_dict())
                
            if current_loss <= self.train_stop_loss_thred:
                break
        
        # Charge le meilleur modèle
        if best_param is not None:
            self.model.load_state_dict(best_param)
            
        self.fitted = True
        return best_loss

    def predict(self, features):
        """Fait des prédictions sur des données numpy"""
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
            
        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(features).to(self.device)
            predictions = self.model(features).cpu().numpy()
            
        return predictions