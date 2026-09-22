"""Download historical trading data from Yahoo Finance."""

import yfinance as yf

# Download data for AAPL (Apple)
print("Downloading AAPL historical data...")
data = yf.download('AAPL', start='2023-01-01', end='2024-12-31', progress=False)

# Save to CSV
filename = 'aapl_historical.csv'
data.to_csv(filename)
print(f"✓ Saved to {filename}")
print(f"\nData shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(f"\nFirst few rows:")
print(data.head())
print(f"\nLast few rows:")
print(data.tail())
