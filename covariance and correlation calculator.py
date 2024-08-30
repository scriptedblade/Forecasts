import yfinance as yf
import pandas as pd

# Define the stock tickers
tickers = ['0P0001PEU9.BO', '0P0001BA07.BO', '0P00005WFD.BO', '0P00005WLZ.BO', '0P0000K9VO.BO', '0P000134CG.BO', '0P0001R7JZ.BO', '0P0001N9FE.BO', '0P00005VCC.BO', '0P00005WFF.BO', '0P00008TPO.BO', '0P00009JAR.BO', '0P0001EQU7.BO', '0P0000AEKG.BO']

# Define the date range
start_date = '2001-01-01'
end_date = '2023-12-31'

# Fetch the historical data
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
daily_returns = data.pct_change().dropna()

# Calculate the covariance matrix
covariance_matrix = daily_returns.cov()

# Calculate the correlation matrix
correlation_matrix = daily_returns.corr()

# Print the results
print("Covariance Matrix:")
print(covariance_matrix)
print("\nCorrelation Matrix:")
print(correlation_matrix)

