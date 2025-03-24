import numpy as np
import pandas as pd
from requests import get
import os
import yfinance as yf

def get_stock_data(ticker): # implements local cache for data
    if os.path.exists(f"stocks/{ticker}.csv"):
        return pd.read_csv(f"stocks/{ticker}.csv")
    
    df = yf.download("AAPL", start="2000-01-01", auto_adjust=True, actions=False, progress=False, multi_level_index=False)
    df["date"] = pd.to_datetime(df.index)
    df["price"] = df["Close"]
    df.drop(labels=["Close", "Open", "High", "Low", "Volume"], axis=1, inplace=True)
    df = df.sort_values(by="date", ascending=True)
    df.reset_index(drop=True, inplace=True)
    df = df.dropna()
    df.to_csv(f"stocks/{ticker}.csv", index=False)
    return df

def SMA(df, period):
    return df['price'].rolling(window=period).mean()

def EMA(df, period):
    return df['price'].ewm(span=period, adjust=False).mean()

def RSI(df, period=14):
    delta = df["price"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def get_full_data(ticker):
    df = get_stock_data(ticker)
    df["SMA-20"] = SMA(df, 20)
    df["SMA-50"] = SMA(df, 50)
    # add column for crossover
    df["RSI"] = RSI(df)
    df["RSI-Momentum"] = df["RSI"] - df["RSI"].shift(1)
    df["Price:SMA"] = (df["price"] - df["SMA-20"])/df["SMA-20"]
    return df

def get_normalized_data(ticker):
    df = get_full_data(ticker)

    df["RSI"] /= 100
    df["RSI-Momentum"] /= 100

    # Manual
    max_price = df["price"].max()
    min_price = df["price"].min()

    def norm_map(values, mn, mx, new_mn, new_mx):
        values -= mn
        values /= (mx - mn)
        values *= (new_mx - new_mn)
        values += new_mn

    norm_map(df["price"], min_price, max_price, 0.01, 1)
    norm_map(df["SMA-20"], min_price, max_price, 0.01, 1)
    norm_map(df["SMA-50"], min_price, max_price, 0.01, 1)


    return df


if __name__ == "__main__":
    print(get_stock_data("MSFT"))