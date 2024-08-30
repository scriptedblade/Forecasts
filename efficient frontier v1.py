import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.optimize import minimize

# Define the stock tickers
tickers = ['HDFCBANK.NS', 'LT.NS', 'RELIANCE.NS', '^NSEI', 'TCS.NS', 'ITC.NS', 'JSWSTEEL.NS', 'ULTRACEMCO.NS', 'NTPC.NS', 'INFY.NS', 'TITAN.NS', 'ONGC.NS']

# Define the date range
start_date = '2001-01-01'
end_date = '2023-12-31'

# Fetch the historical data
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
daily_returns = data.pct_change().dropna()

# Calculate expected returns (mean of daily returns)
expected_returns = daily_returns.mean()

# Calculate the covariance matrix of daily returns
covariance_matrix = daily_returns.cov()

# Calculate skewness for each asset
asset_skewness = daily_returns.apply(skew)
print("\nSkewness of each asset:")
print(asset_skewness)

# Number of portfolios to simulate
num_portfolios = 90000

# Initialize arrays to hold portfolio returns, risk, weights, Sharpe Ratios, and skewness
portfolio_returns = []
portfolio_risks = []
portfolio_weights = []
sharpe_ratios = []
portfolio_skewness = []

# Risk-free rate (annual)
annual_risk_free_rate = 0.065  # Assuming 6.5% annual risk-free rate
daily_risk_free_rate = annual_risk_free_rate / 252

# Simulate random portfolios
np.random.seed(42)
for _ in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    returns = np.dot(weights, expected_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe_ratio = (returns - daily_risk_free_rate) / risk
    
    # Calculate the portfolio returns for each day
    portfolio_daily_returns = (daily_returns * weights).sum(axis=1)
    
    # Calculate skewness of the portfolio returns
    portfolio_skew = skew(portfolio_daily_returns)
    
    portfolio_returns.append(returns)
    portfolio_risks.append(risk)
    portfolio_weights.append(weights)
    sharpe_ratios.append(sharpe_ratio)
    portfolio_skewness.append(portfolio_skew)

# Convert results to DataFrame
portfolio_results = pd.DataFrame({
    'Returns': portfolio_returns,
    'Risk': portfolio_risks,
    'Sharpe Ratio': sharpe_ratios,
    'Skewness': portfolio_skewness
})

# Annualize the returns and risk
portfolio_results['Annualized Returns'] = portfolio_results['Returns'] * 252
portfolio_results['Annualized Risk'] = portfolio_results['Risk'] * np.sqrt(252)

# Find the portfolio with the minimum risk
min_risk_idx = portfolio_results['Risk'].idxmin()
min_risk_portfolio = portfolio_results.loc[min_risk_idx]
min_risk_weights = portfolio_weights[min_risk_idx]

# Print weights and returns of the portfolio with minimum risk
print("\nWeights of Portfolio with Minimum Risk:")
for ticker, weight in zip(tickers, min_risk_weights):
    print(f"{ticker}: {weight:.4f}")

# Print risk and return of the portfolio with minimum risk
min_risk_return = min_risk_portfolio['Returns'] * 252  # Annualizing the return
min_risk_value = min_risk_portfolio['Risk'] * np.sqrt(252)  # Annualizing the risk
min_risk_sharpe_ratio = (min_risk_return - annual_risk_free_rate) / min_risk_value
print(f"\nMaximum Return for the Least Risk: {min_risk_return:.2%}")
print(f"Risk Associated with the Minimum Risk Portfolio: {min_risk_value:.2%}")
print(f"Sharpe Ratio of the Minimum Risk Portfolio: {min_risk_sharpe_ratio:.4f}")
print(f"Skewness of the Minimum Risk Portfolio: {min_risk_portfolio['Skewness']:.4f}")

# Mean-Variance Optimization (MVO) to find the optimal portfolio
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (returns - risk_free_rate) / std
    return std, returns, sharpe_ratio

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

optimal_portfolio = optimize_portfolio(expected_returns, covariance_matrix, daily_risk_free_rate)

optimal_weights = optimal_portfolio.x
optimal_std, optimal_return, optimal_sharpe = portfolio_performance(optimal_weights, expected_returns, covariance_matrix, daily_risk_free_rate)
optimal_annual_return = optimal_return * 252
optimal_annual_risk = optimal_std * np.sqrt(252)

# Calculate skewness of the optimal portfolio
optimal_portfolio_daily_returns = (daily_returns * optimal_weights).sum(axis=1)
optimal_portfolio_skewness = skew(optimal_portfolio_daily_returns)

print("\nOptimal Portfolio Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

print(f"\nOptimal Portfolio Annualized Return: {optimal_annual_return:.2%}")
print(f"Optimal Portfolio Annualized Risk: {optimal_annual_risk:.2%}")
print(f"Optimal Portfolio Sharpe Ratio: {optimal_sharpe:.4f}")
print(f"Skewness of the Optimal Portfolio: {optimal_portfolio_skewness:.4f}")

# Plot the Efficient Frontier with annualized returns and risks
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_results['Annualized Risk'], portfolio_results['Annualized Returns'], c=portfolio_results['Sharpe Ratio'], cmap='viridis', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(min_risk_portfolio['Annualized Risk'], min_risk_portfolio['Annualized Returns'], color='blue', marker='*', s=200, label='Min Risk')
plt.scatter(optimal_annual_risk, optimal_annual_return, color='red', marker='*', s=200, label='Optimal Portfolio')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.legend()
plt.show()
