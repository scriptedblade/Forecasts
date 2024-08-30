import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

# Monte Carlo Simulation
num_portfolios = 10000
results = np.zeros((3 + len(tickers), num_portfolios))
np.random.seed(42)

for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    
    # Expected portfolio return
    portfolio_return = np.sum(weights * daily_returns.mean()) * 252  # Annualizing the return
    
    # Expected portfolio risk
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix * 252, weights)))  # Annualizing the risk
    
    # Store the results
    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    results[2,i] = results[0,i] / results[1,i]  # Sharpe Ratio
    results[3:,i] = weights

# Convert results array to Pandas DataFrame
columns = ['Return', 'Risk', 'Sharpe Ratio'] + tickers
results_frame = pd.DataFrame(results.T, columns=columns)

# Locate the portfolio with the highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['Sharpe Ratio'].idxmax()]

# Locate the portfolio with the minimum risk
min_risk_port = results_frame.iloc[results_frame['Risk'].idxmin()]

# Plotting the Monte Carlo simulation results
plt.scatter(results_frame.Risk, results_frame.Return, c=results_frame['Sharpe Ratio'], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(max_sharpe_port['Risk'], max_sharpe_port['Return'], marker='*', color='r', s=500, label='Max Sharpe Ratio')
plt.scatter(min_risk_port['Risk'], min_risk_port['Return'], marker='*', color='b', s=500, label='Min Risk')
plt.title('Monte Carlo Simulation for Portfolio Optimization')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.legend()
plt.show()

# Function to minimize negative Sharpe ratio
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return -sharpe_ratio

# Function to get portfolio return and risk
def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_std_dev

# Get mean returns and covariance of returns
mean_returns = daily_returns.mean() * 252
cov_matrix = daily_returns.cov() * 252

# Generate the efficient frontier
target_returns = np.linspace(results_frame['Return'].min(), results_frame['Return'].max(), 100)
efficient_frontier_weights = []

for target_return in target_returns:
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: np.sum(x * mean_returns) - target_return})
    bounds = tuple((0, 1) for asset in range(len(tickers)))
    result = minimize(negative_sharpe_ratio, len(tickers) * [1. / len(tickers)], args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        efficient_frontier_weights.append(result.x)

# Plot the efficient frontier
efficient_frontier_risks = [portfolio_performance(weights, mean_returns, cov_matrix)[1] for weights in efficient_frontier_weights]
plt.plot(efficient_frontier_risks, target_returns, 'r--', linewidth=3)
plt.scatter(results_frame.Risk, results_frame.Return, c=results_frame['Sharpe Ratio'], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(max_sharpe_port['Risk'], max_sharpe_port['Return'], marker='*', color='r', s=500, label='Max Sharpe Ratio')
plt.scatter(min_risk_port['Risk'], min_risk_port['Return'], marker='*', color='b', s=500, label='Min Risk')
plt.title('Monte Carlo Simulation with Efficient Frontier')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.legend()
plt.show()

# Print details of the optimal portfolios
print("\nPortfolio with the maximum Sharpe Ratio:\n")
print(max_sharpe_port)

print("\nPortfolio with the minimum risk:\n")
print(min_risk_port)
