import main
from main import preprocess_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, dataset
import numpy as np

win_size = 30
batch_size = 32
epochs = 30
lr = 0.001

train, val, test = main.preprocess_data('APPL')


device = 'mps' if torch.cuda.is_available() else 'cpu'
print(f"device:{device}")

def create_sequences(data, win_size):
    X, y =[], []
    for i in range(len(data)-win_size):
        X.append(data[i:i+win_size])
        y.append(data[i+win_size])
    return np.array(X), np.array(y)

class StockDataset(Dataset):
    def __init__(self, X,y):
        self.X = torch.tensor(X, dtype= torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class StockLSTM(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 50, num_layers = 2):
        super(StockLSTM, self).__init__() #상속한 nn.Module의 초기화 메서드 호출
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x ):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
X_train ,y_train= create_sequences(train,win_size)
X_test ,y_test = create_sequences(test, win_size)

train_loader = torch.utils.data.Dataloader(StockDataset(X_train, y_train))