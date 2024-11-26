import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from statsmodels.tsa.stattools import adfuller

# Step 1: Generate synthetic housing prices data
np.random.seed(42)
dates = pd.date_range('2015-01-01', '2024-11-01', freq='M')
prices = np.random.normal(loc=350000, scale=50000, size=len(dates)) + np.linspace(0, 100000, len(dates))

# Create a DataFrame
sample_data = pd.DataFrame({'date': dates, 'price': prices})
sample_data.set_index('date', inplace=True)

# Save to CSV
file_path = '/mnt/data/vienna_housing_prices.csv'
sample_data.to_csv(file_path)

# Display the first few rows of the dataset
print("Dataset preview:")
print(sample_data.head())

# Step 2: Visualize the historical data
plt.figure(figsize=(10,6))
plt.plot(sample_data.index, sample_data['price'], label='Housing Prices')
plt.title('Housing Prices in Vienna Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Step 3: Check for stationarity (ADF test)
result = adfuller(sample_data['price'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# If p-value > 0.05, the data is non-stationary, and we may need differencing (ARIMA model)

# Step 4: Fit the ARIMA model
model = ARIMA(sample_data['price'], order=(5, 1, 0))  # AR(5), I(1), MA(0) as an example
model_fit = model.fit()

# Summary of the model
print("\nARIMA Model Summary:")
print(model_fit.summary())

# Step 5: Forecast future housing prices (next 3 years)
forecast_steps = 36  # Forecasting 36 months (3 years)
forecast = model_fit.forecast(steps=forecast_steps)

# Generate future dates
future_dates = pd.date_range(sample_data.index[-1], periods=forecast_steps+1, freq='M')[1:]

# Plot the forecasted data
plt.figure(figsize=(10,6))
plt.plot(sample_data.index, sample_data['price'], label='Historical Prices')
plt.plot(future_dates, forecast, label='Forecasted Prices', color='red')
plt.title('Forecasted Housing Prices in Vienna for Next 3 Years')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Print forecasted values
forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Forecasted Price'])
print("\nForecasted Housing Prices for the Next 3 Years:")
print(forecast_df)

# Save the forecasted data to a CSV file
forecast_file_path = '/mnt/data/forecasted_vienna_housing_prices.csv'
forecast_df.to_csv(forecast_file_path)

# Provide the link to download the forecasted data
print(f"\nForecasted data saved to: {forecast_file_path}")
