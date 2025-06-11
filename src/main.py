import pandas as pd
import yfinance as yf
import numpy as np
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
for ticker in stock_ticker:
    df = yf.download(ticker, start=start_, end=end_ )

    #csv로 저장
    df.to_csv(f'data/{ticker}_data.csv', index=True)
    print(f"{ticker}주가 저장완료") 

# df['Close'].plot(kind='line', figsize=(8,4),title='appl Stock close ')
# plt.show()
# plt.savefig('AAPL_stock_close.png')

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
#0~1 사이로 정규화
def preprocess_data(ticker):
    df = pd.read_csv(f"data/{ticker}_data.csv", header=[0, 1]) # 2중 헤더가 있는 CSV 파일을 읽어옴 (첫 번째 줄: 데이터 종류, 두 번째 줄: AAPL 등 Ticker)
    df.columns = df.columns.droplevel(0) # 첫 번째 헤더 레벨(예: 'Price', 'Close', ...) 제거 → 실제 값이 있는 Ticker만 남김
    df = df.rename(columns={'AAPL': 'Close'}) # 종가 데이터가 들어있는 첫 번째 'AAPL' 컬럼명을 'Close'로 명시적으로 변경
    df = df.rename_axis("Date").reset_index()# 인덱스가 없을 경우를 대비해 'Date'라는 이름을 붙이고, 열로 되돌림
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')# 'Date' 열을 DataFrame의 인덱스로 설정 → 시계열 데이터로 다루기 위함

    df= df[['Close','High','Low','Open','volume']]
    df = df.dropna()

    prices = df[['Close']].values #종가만 사용

    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)

    #데이터 분할 (시계열이라 셔플 X)
    train_size = int(len(prices_scaled) * 0.8)
    val_size = int(len(prices_scaled) * 0.1)

    train = prices_scaled[:train_size]
    val = prices_scaled[train_size:train_size + val_size]
    test = prices_scaled[train_size + val_size:]

    np.save(f'data/{ticker}_train.npy', train)
    np.save(f'data/{ticker}_val.npy', val)
    np.save(f'data/{ticker}_test.npy', test)

    print(f"{ticker}데이터 전처리 완료")
    print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test

preprocess_data('AAPL')