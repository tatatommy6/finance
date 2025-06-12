import main
from main import preprocess_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
device = 'mps' if torch.cuda.is_available() else 'cpu'
print(f"device:{device}")

X_train_tensors = torch.Tensor(preprocess_data.X_train)
X_train_tensors = torch.Tensor(preprocess_data.X_train)

y_train_tensors = torch.Tensor(preprocess_data.y_train)
X_test_tensors = torch.Tensor(preprocess_data.X_test)

X_train_tensors_f = torch.reshape(X_train_tensors, X_train_tensors.shape[0], 1, X_train_tensors.shape[1])
X_test_tensors_f = torch.reshape(X_test_tensors, X_test_tensors.shape[0], 1, X_test_tensors.shape[1])

print()

class StockLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(StockLSTM, self).__init__() #상속한 nn.Module의 초기화 메서드 호출
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        # LSTM 레이어 정의
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x ):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.jidden_size).requires_grad_()
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))  # LSTM 레이어 통과
        #최종 은닉 상태를 완전 연결층에 입력하기 위해 2차원 텐서로 변환
        hn = hn[-1] #최종 lstm 층의 마지막 은닉 상태만 사용함

        out = self.relu(hn)
        out = self.fc_1(out)  # 완전 연결층 통과
        out = self.relu(out)

        out = self.fc(out)  # 최종 출력층 통과

        return out
    
num_epochs = 10000
lr = 0.0001

input_size = 5
hidden_size = 64
num_layers = 1
num_classes = 1

model = StockLSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_f.shape[1])

# 모델 저장
torch.save(model.state_dict(), 'model/stock_lstm.pth')