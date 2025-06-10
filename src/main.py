import pandas as pd
import yfinance as yf
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

df['Close'].plot(kind='line', figsize=(8,4),title='appl Stock close ')
plt.show()
plt.savefig('AAPL_stock_close.png')
