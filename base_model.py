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

    def fit(self, dl_train, dl_valid):
        train_loader = self._init_data_loader(dl_train, shuffle=True)
        valid_loader = self._init_data_loader(dl_valid, shuffle=False)
        
        self.fitted = True
        best_param = None
        best_valid_loss = float('inf')
        
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            valid_loss = self.test_epoch(valid_loader)
            
            print(f"Epoch {step}, train_loss {train_loss:.6f}, valid_loss {valid_loss:.6f}")
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_param = copy.deepcopy(self.model.state_dict())
            
            if train_loss <= self.train_stop_loss_thred:
                break
                
        # Sauvegarde du meilleur modèle
        torch.save(
            best_param,
            f'{self.save_path}{self.save_prefix}crypto_master_{self.seed}.pkl'
        )
        
        # Charge le meilleur modèle pour la suite
        self.model.load_state_dict(best_param)

    def predict(self, dl_test):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
            
        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)
        
        preds = []
        ic = []
        ric = []
        
        self.model.eval()
        with torch.no_grad():
            for data in test_loader:
                data = torch.squeeze(data, dim=0)
                feature = data[:, :, :-1].to(self.device)
                label = data[:, -1, -1]
                
                pred = self.model(feature.float()).cpu().numpy()
                preds.append(pred.ravel())
                
                daily_ic, daily_ric = calc_ic(pred, label.numpy())
                ic.append(daily_ic)
                ric.append(daily_ric)
                
        predictions = pd.Series(
            np.concatenate(preds), 
            index=dl_test.get_index()
        )
        
        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric)
        }
        
        return predictions, metrics