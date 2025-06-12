import pandas as pd
import yfinance as yf
import numpy as np
import datetime
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


stock_ticker = ['AAPL']
start_ = '2022-06-09'
end_ = '2025-06-09'
torch.manual_seed(42) #랜덤 시드 고정
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42) #GPU 사용시 랜덤 시드 고정
for ticker in stock_ticker:
    df = yf.download(ticker, start=start_, end=end_ )

    #csv로 저장
    df.to_csv(f'data/{ticker}_data.csv', index=True)
    print(f"{ticker}주가 저장완료") 

# df['Close'].plot(kind='line', figsize=(8,4),title='appl Stock close ')
# plt.show()
# plt.savefig('AAPL_stock_close.png')



# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#0~1 사이로 정규화
X = df.drop('Close', axis=1)
y = df[['Close']]
ms = MinMaxScaler()
ss = StandardScaler()
X_ss = ss.fit_transform(X)
y_ms = ms.fit_transform(y)

X_train = X_ss[:int(len(X_ss)*0.7), :]
X_test = X_ss[int(len(X_ss)*0.7):, :]

y_train = y_ms[:int(len(y_ms)*0.7), :]
y_test = y_ms[int(len(y_ms)*0.7):, :]

print('training shape : ', X_train.shape, y_train.shape)
print('Testing shape : ', X_test.shape, y_test.shape)

X_train_tensors = torch.Tensor(X_train)
X_train_tensors = torch.Tensor(X_train)

y_train_tensors = torch.Tensor(y_train)
X_test_tensors = torch.Tensor(X_test)

X_train_tensors_f = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_f = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))


print("train shape:", X_train.shape, y_train.shape)
print("test shape:", X_test.shape, y_test.shape)

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
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h_0, c_0))  # LSTM 레이어 통과
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

crierterion = torch.nn.MSELoss() #손실함수
optimizer = torch.optim.Adam(model.parameters(), lr )

for eopch in range(num_epochs):
    outputs = model(X_train_tensors_f)  # 모델을 학습 모드로 설정
    optimizer.zero_grad()  # 이전 기울기 초기화
    loss = crierterion(outputs, y_train_tensors)  # 손실 계산
    loss.backward()  # 역전파 수행

    optimizer.step()  # 가중치 업데이트
    if (eopch+1) % 100 == 0:  # 100 에폭마다 손실 출력
        print(f'Epoch [{eopch+1}/{num_epochs}], Loss: {loss.item():1.5f}')

df_x_ss = ss.transform(X)
df_y_ms = ms.transform(y)

df_x_ss = torch.Tensor(df_x_ss, dtype = torch.float32)
df_y_ms = torch.Tensor(df_y_ms, dtype = torch.float32)

df_x_ss = df_x_ss.unsqueeze(1)

train_predict = model(df_x_ss)
# 텐서를 넘파이 배열로 변환
predicted = train_predict.detach().numpy()  # 기울기 추적 비활성화
label_y = df_y_ms.detach().numpy()  

# 예측값과 실제값을 원래 스케일로 되돌림
predicted = ms.inverse_transform(predicted)
label_y = ms.inverse_transform(label_y)

# 결과 시각화
plt.figure(figsize=(10, 6))

# 훈련/테스트 구분 날짜에 빨간 세로선 추가
plt.axvline(x=datetime(2024, 4, 1), color='r', linestyle='--')

# 데이터프레임에 예측값 추가
df['pred'] = predicted

# 실제 데이터와 예측 데이터를 플롯
plt.plot(df['Close'], label='Actual Data')
plt.plot(df['pred'], label='Predicted Data')

# 그래프 제목과 범례 설정
plt.title('Time-series Prediction')
plt.legend()
plt.show()

# 모델 저장
torch.save(model.state_dict(), 'model/stock_lstm.pth')