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

