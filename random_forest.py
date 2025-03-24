import stock_data
import pandas as pd
import numpy as np

tickers = [
    # The classics
    "SPY", "VTV", "VTG", "QQQ", "SCHD",

    # Blue-chip stocks
    "JNJ", "PG", "KO", "PEP", "WMT", "MCD",
    
    # Low-volatility ETFs
    "USMV", "SPLV", "VIG", "SCHD", "VTV",
    
    # Defensive sector ETFs
    "XLU", "XLV", "XLP",

    "DGRW", "VYM", "HDV", "NOBL", "DGRO", #"FDL", "DVY", "SPHD", "VPU", "FSTA", "FUTY", "XLU", "IDU", "RHS", "VDC", "IYK", "PFXF", "PFFD", "PSK", "LDRS", "LVHD", "QDF", "FTSD", "TLO", "SPLB", "SPBO", "BND", "AGG", "SHY", "TIP"
]

df = pd.DataFrame()
for ticker in tickers:
    sub_df = stock_data.get_normalized_data(ticker)
    sub_df["ticker"] = ticker
    df = pd.concat([df, sub_df], axis=0)

# Set target
# df["Target"] = df['price'].shift(-1) # regression
df = df.dropna()
df['Target'] = (df['price'].shift(-1) > df['price']).astype(int) # classification


X = df.copy()
X = X.drop(labels=["Target", "ticker"], axis=1)
X["date"] = pd.to_datetime(X["date"]).astype('int64')

y = df["Target"].copy()

split_index = int(len(df) * 0.8)
X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

def simulate_portfolio(prices, signals, initial_balance=10000):
    """
    Simulates portfolio balance over time based on buy/sell signals.

    Parameters:
        prices (pd.Series): Series of actual prices (e.g., closing price).
        signals (list or Series): List of 'buy' or 'sell' signals (must align with prices).
        initial_balance (float): Starting portfolio value.

    Returns:
        pd.DataFrame: Portfolio balance, holdings, and action log over time.
    """
    df = pd.DataFrame({
        "price": prices.values,
        "signal": signals
    }).reset_index(drop=True)
    print(df["price"])

    balance = initial_balance
    holdings = 0
    portfolio_values = []

    for i in range(len(df) - 1):  # use next day's price
        today_signal = df.loc[i, "signal"]
        next_price = df.loc[i + 1, "price"]

        if today_signal == "buy" and holdings == 0:
            # Buy with full balance
            holdings = balance / next_price
            try:
                assert(not np.isnan(holdings))
            except AssertionError:
                print(holdings, balance, next_price)
                quit()
            balance = 0
            action = "BUY"
        elif today_signal == "sell" and holdings > 0:
            # Sell all holdings
            balance = holdings * next_price
            holdings = 0
            action = "SELL"
        else:
            action = "HOLD"

        portfolio_value = balance + holdings * next_price
        print(holdings)
        portfolio_values.append({
            "day": i,
            "price": next_price,
            "signal": today_signal,
            "action": action,
            "holdings": holdings,
            "balance": balance,
            "portfolio_value": portfolio_value
        })

    return pd.DataFrame(portfolio_values)

def classify_signal_with_hold(probs, threshold=0.6):
    """
    Converts class probabilities into 'buy', 'sell', or 'hold' signals.
    - probs: Nx2 numpy array from model.predict_proba(X)
    - threshold: minimum probability to trigger a buy/sell
    """
    signals = []

    for prob_down, prob_up in probs:
        if prob_up >= threshold:
            signals.append("buy")
        elif prob_down >= threshold:
            signals.append("sell")
        else:
            signals.append("hold")

    return signals
signals = classify_signal_with_hold(model.predict_proba(X_test), )

portfolio_df = simulate_portfolio(prices=X_test["price"], signals=signals)

import matplotlib.pyplot as plt
plt.plot(portfolio_df["portfolio_value"])
plt.title("Simulated Portfolio Value Over Time")
plt.xlabel("Weeks")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.show()

# Optional: print final value
print(f"Final Portfolio Value: ${portfolio_df['portfolio_value'].iloc[-1]:.2f}")