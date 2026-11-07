import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import requests
from io import StringIO
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(123)

# PART 1: VOLATILITY MODELING FROM VIX DATA

print("\nPART 1: VOLATILITY MODELING FROM VIX DATA\n")

# Download VIX data
url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
response = requests.get(url)
vix_data = pd.read_csv(StringIO(response.text))

# Convert DATE column to datetime format
vix_data['DATE'] = pd.to_datetime(vix_data['DATE'])

# Sort by date and get one year of data
vix_data = vix_data.sort_values('DATE')
end_date = vix_data['DATE'].max()
start_date = end_date - timedelta(days=365)
vix_one_year = vix_data[(vix_data['DATE'] >= start_date) & (vix_data['DATE'] <= end_date)]

print("Date range for VIX analysis:")
print(f"Start: {vix_one_year['DATE'].min().date()}")
print(f"End: {vix_one_year['DATE'].max().date()}")
print(f"Number of observations: {len(vix_one_year)}\n")

# Calculate log returns
vix_one_year['log_return'] = np.log(vix_one_year['CLOSE']).diff()
log_returns = vix_one_year['log_return'].dropna().values

# Summary statistics
print("Summary statistics of VIX log returns:")
print(f"Mean: {np.mean(log_returns):.6f}")
print(f"Median: {np.median(log_returns):.6f}")
print(f"Min: {np.min(log_returns):.6f}")
print(f"Max: {np.max(log_returns):.6f}")
print(f"Standard deviation: {np.std(log_returns, ddof=1):.6f}")
print(f"Skewness: {stats.skew(log_returns):.6f}")
print(f"Kurtosis: {stats.kurtosis(log_returns, fisher=False):.6f}\n")

# Fit Student's t-distribution
def fit_t_distribution(data):
    """Fit Student's t-distribution to data"""
    df, loc, scale = stats.t.fit(data)
    return df, loc, scale

df_est, m_est, s_est = fit_t_distribution(log_returns)

print("Student's t-distribution parameters:")
print(f"Degrees of freedom (df): {df_est:.6f}")
print(f"Location (m): {m_est:.6f}")
print(f"Scale (s): {s_est:.6f}\n")

# PART 2: SETUP FIXED GRID AND PARAMETERS

print("\nPART 2: FIXED GRID SETUP\n")

# Fixed parameters
base_strike = 100
strike_scalars = [0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40]
strike_prices = [base_strike * scalar for scalar in strike_scalars]

maturities = list(range(1, 6))  # 1 to 5 years

# Create fixed grid
fixed_grid = pd.DataFrame(
    [(K, T) for K in strike_prices for T in maturities],
    columns=['K', 'T']
)

print(f"Fixed Grid:")
print(f"Strike prices: {strike_prices}")
print(f"Maturities (years): {maturities}")
print(f"Total grid points: {len(fixed_grid)}\n")

# PART 2.5: FETCH US 1-YEAR TREASURY RATE (PAST 6 MONTHS AVERAGE)

print("\nPART 2.5: FETCHING US 1-YEAR TREASURY RATE\n")

# Download DGS1 (1-year Treasury rate) from FRED
print("Downloading 1-Year Treasury Rate data from FRED...")
treasury_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1"
treasury_response = requests.get(treasury_url)
treasury_data = pd.read_csv(StringIO(treasury_response.text))

# Convert to proper format
treasury_data.columns = ['Date', 'Rate']
treasury_data['Date'] = pd.to_datetime(treasury_data['Date'])
treasury_data['Rate'] = pd.to_numeric(treasury_data['Rate'], errors='coerce')

# Remove NA values
treasury_data = treasury_data.dropna()

# Get the past 6 months of data
end_date_treasury = treasury_data['Date'].max()
start_date_treasury = end_date_treasury - timedelta(days=180)

treasury_6m = treasury_data[
    (treasury_data['Date'] >= start_date_treasury) & 
    (treasury_data['Date'] <= end_date_treasury)
]

print(f"Treasury rate data range:")
print(f"Start: {treasury_6m['Date'].min().date()}")
print(f"End: {treasury_6m['Date'].max().date()}")
print(f"Number of observations: {len(treasury_6m)}\n")

# Calculate average rate (convert from percentage to decimal)
risk_free_rate = treasury_6m['Rate'].mean() / 100

print("1-Year Treasury Rate Statistics (past 6 months):")
print(f"Mean: {treasury_6m['Rate'].mean():.2f}%")
print(f"Median: {treasury_6m['Rate'].median():.2f}%")
print(f"Min: {treasury_6m['Rate'].min():.2f}%")
print(f"Max: {treasury_6m['Rate'].max():.2f}%")
print(f"Standard Deviation: {treasury_6m['Rate'].std():.2f}%\n")

print(f"Risk-free rate for simulation: {risk_free_rate:.4f} ({risk_free_rate*100:.2f}%)\n")

# Simulation parameters
n_scenarios = 50000
n_months = 12
initial_price = 100  # Starting price of underlying asset

print("Simulation parameters:")
print(f"Number of scenarios: {n_scenarios}")
print(f"Number of months: {n_months}")
print(f"Risk-free rate: {risk_free_rate:.4f} ({risk_free_rate*100:.2f}%)")
print(f"Initial asset price: {initial_price}\n")

# PART 3: BLACK-SCHOLES FORMULA

def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes formula for European call option
    S: current stock price
    K: strike price
    T: time to maturity (years)
    r: risk-free rate
    sigma: volatility
    """
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    
    return call_price

# PART 4: MONTE CARLO SIMULATION

print("\nPART 3: RUNNING MONTE CARLO SIMULATIONS\n")

# Initialize results list
results_list = []

# Price simulation parameters (Geometric Brownian Motion)
mu = 0.08  # Expected annual return
dt = 1/12  # Monthly time step

print("Starting Monte Carlo simulation...")
print("This may take a few moments...\n")

# Run simulations for each month
for month in range(1, n_months + 1):
    
    if month % 3 == 0:
        print(f"Processing month {month} of {n_months}...")
    
    # Simulate volatility from fitted t-distribution
    volatility_shocks = stats.t.rvs(df=df_est, loc=m_est, scale=s_est, size=n_scenarios)
    
    # Convert log returns to volatility levels (annualized)
    base_volatility = vix_one_year['CLOSE'].mean() / 100  # VIX is in percentage points
    simulated_volatility = np.maximum(base_volatility * np.exp(volatility_shocks), 0.01)
    
    # Simulate stock prices using GBM
    Z = np.random.randn(n_scenarios)
    simulated_prices = initial_price * np.exp(
        (mu - 0.5 * simulated_volatility**2) * month * dt + 
        simulated_volatility * np.sqrt(month * dt) * Z
    )
    
    # For each grid point, calculate option prices
    for idx, row in fixed_grid.iterrows():
        K = row['K']
        T_maturity = row['T']
        
        # Time remaining = original maturity - time elapsed
        T_remaining = max(T_maturity - month/12, 0.001)
        
        # Calculate call prices for all scenarios
        call_prices = black_scholes_call(
            S=simulated_prices,
            K=K,
            T=T_remaining,
            r=risk_free_rate,
            sigma=simulated_volatility
        )
        
        # Create temporary dataframe for this grid point and month
        temp_df = pd.DataFrame({
            'simulation_number': range(1, n_scenarios + 1),
            'month': month,
            'S': simulated_prices,
            'K': K,
            'T': T_remaining,
            'sigma': simulated_volatility,
            'call_price': call_prices
        })
        
        results_list.append(temp_df)

# Combine all results
results = pd.concat(results_list, ignore_index=True)

print("\nSimulation complete!")
print(f"Total number of rows in results: {len(results)}")
print("\nResults stored in 'results' DataFrame")
print(f"DataFrame shape: {results.shape}")
print(f"\nFirst few rows:")
print(results.head())
print(f"\nDataFrame info:")
print(results.info())