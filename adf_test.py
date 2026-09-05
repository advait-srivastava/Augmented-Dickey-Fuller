import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf

# Download stock data for Apple (AAPL)
stock_data = yf.download("AAPL", start="2020-01-01", end="2023-12-31")
stock_prices = stock_data['Close']  # Use the closing prices

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(stock_prices, label='AAPL Closing Prices')
plt.title("Apple Stock Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
from statsmodels.tsa.stattools import adfuller

# Perform ADF test
result = adfuller(stock_prices)
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("The series is stationary.")
else:
    print("The series is not stationary. Differencing is needed.")
