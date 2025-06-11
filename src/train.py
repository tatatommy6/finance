import main
from main import preprocess_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

win_size = 30
batch_size_ = 32
epochs = 30
lr_ = 0.001

train = np.load('data/AAPL_train.npy')
val = np.load('data/AAPL_val.npy')
test = np.load('data/AAPL_test.npy')


device = 'mps' if torch.cuda.is_available() else 'cpu'
print(f"device:{device}")
print(np.min(train), np.max(train))

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
    def __init__(self, input_size = 5, hidden_size = 50, num_layers = 2):
        super(StockLSTM, self).__init__() #상속한 nn.Module의 초기화 메서드 호출
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 5)

    def forward(self, x ):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
X_train ,y_train= create_sequences(train,win_size)
X_val ,y_val = create_sequences(val, win_size)

print("Train shape:", X_train.shape, y_train.shape)
print("Val shape:", X_val.shape, y_val.shape)
print("Sample input:", X_train[0])
print("Sample target:", y_train[0])

train_loader = torch.utils.data.DataLoader(StockDataset(X_train, y_train), batch_size = batch_size_, shuffle = False)
val_loader = torch.utils.data.DataLoader(StockDataset(X_val, y_val),batch_size = batch_size_, shuffle = False)

model = StockLSTM()
cirterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr_)

print(len(val_loader))

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()
        output = model(X_batch)
        loss = cirterion(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_loss = len(train_loader)
    val_loss = 0

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            loss = cirterion(output, y_batch)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Train Loss : {epoch_loss/len(train_loader):.4f} Val Loss: {val_loss/len(val_loader):.4f}")

# 모델 저장
torch.save(model.state_dict(), 'model/stock_lstm.pth')