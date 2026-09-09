"""Task 1 - Investigate and analyse natural gas price data (JPMorgan Forage).

Loads a monthly natural gas price series, plots it, fits a Holt-Winters
``ExponentialSmoothing`` model (additive trend, additive seasonality,
``seasonal_periods=12``) and forecasts the next 12 month-end prices. The
``estimate_price(date)`` helper returns the historical price for a date in the
input series, the forecast price for a date in the 12-month forecast window,
and an out-of-range message otherwise - it requires an exact month-end index
match rather than interpolating.

Input: ``pd.read_csv("Nat_Gas.csv")`` with ``Date`` and ``Price`` columns; the
path is hard-coded and relative to the working directory. The bundled series is
``datasets/naturalgas.csv`` (48 monthly observations from 10/31/20), but its
columns are named ``Dates`` and ``Prices``, so both the filename and the column
names need updating before the script will run.

Output: two matplotlib figures (price history, and history plus forecast) shown
interactively, and the estimated price for the hard-coded date ``2025-06-30``
printed to stdout.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the data
file_path = "Nat_Gas.csv"  # Update this with the actual file path
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime format
df.set_index('Date', inplace=True)

# Sort data by date in case it's not sorted
df = df.sort_index()

# Visualize the data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Price'], marker='o', linestyle='-', label='Natural Gas Price')
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Natural Gas Price Trend")
plt.legend()
plt.grid()
plt.show()

# Apply Exponential Smoothing for forecasting
model = ExponentialSmoothing(df['Price'], trend='add', seasonal='add', seasonal_periods=12)
fit_model = model.fit()

# Forecast for the next 12 months (1 year)
future_dates = pd.date_range(start=df.index[-1], periods=13, freq='M')[1:]
future_forecast = fit_model.forecast(12)

# Create forecast DataFrame
forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Price': future_forecast})
forecast_df.set_index('Date', inplace=True)

# Plot historical data and forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Price'], marker='o', linestyle='-', label='Historical Price')
plt.plot(forecast_df.index, forecast_df['Forecasted_Price'], marker='o', linestyle='--', label='Forecasted Price')
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Natural Gas Price Forecast")
plt.legend()
plt.grid()
plt.show()

# Function to estimate price for a given date
def estimate_price(date):
    date = pd.to_datetime(date)
    if date in df.index:
        return df.loc[date, 'Price']
    elif date in forecast_df.index:
        return forecast_df.loc[date, 'Forecasted_Price']
    else:
        return "Date out of range. Please provide a date within the given range."

# Example usage
input_date = "2025-06-30"  # Change this to the desired date
print(f"Estimated price on {input_date}: {estimate_price(input_date)}")

