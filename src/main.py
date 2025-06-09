import pandas as pd
import yfinance as yf

stock_ticker = ['AAPL','NVDA','GOOG']
start_ = '2022-06-09'
end_ = '2025-06-09'
for ticker in stock_ticker:
    df = yf.download(ticker, start=start_, end=end_ )

    #csv로 저장
    df.to_csv(f'data/{ticker}_data.csv', index=True)
    print(f"{ticker}주가 저장완료") 