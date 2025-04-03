import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
file_path = "data (2).csv"
df = pd.read_csv(file_path)

# Convert 'date' to datetime format and sort values
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# Forward fill missing values, then backward fill any remaining
# This ensures no NaN values remain
df.ffill(inplace=True)
df.bfill(inplace=True)

# Set 'date' as the index and assign frequency
df.set_index('date', inplace=True)
df = df.asfreq('D')  # Set daily frequency

# Function to perform Augmented Dickey-Fuller (ADF) test
def adf_test(series, title=""):
    try:
        result = adfuller(series.dropna())  # Ensure no NaN values
        print(f"\n### Augmented Dickey-Fuller Test ({title}) ###")
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        if result[1] <= 0.05:
            print("✅ Data is stationary (p-value ≤ 0.05), proceed with modeling")
        else:
            print("❌ Data is NOT stationary (p-value > 0.05), apply differencing")
    except Exception as e:
        print(f"Error in ADF test: {e}")

# Initial ADF test on 'last_value'
adf_test(df['last_value'], title="Original Data")

# Apply first-order differencing
df['diff_last_value'] = df['last_value'].diff()

# ADF test after first-order differencing
adf_test(df['diff_last_value'], title="First-Order Differenced Data")

# Plot first-order differenced data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['diff_last_value'], label="Differenced Closing Price", color='orange')
plt.xlabel("Date")
plt.ylabel("Differenced Closing Price")
plt.title("First-Order Differenced Stock Price Over Time")
plt.legend()
plt.grid()
plt.show()

# If still not stationary, apply second-order differencing
if adfuller(df['diff_last_value'].dropna())[1] > 0.05:
    df['diff_last_value_2'] = df['diff_last_value'].diff()
    adf_test(df['diff_last_value_2'], title="Second-Order Differenced Data")

# Plot ACF and PACF to determine p and q values
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(df['diff_last_value'].dropna(), ax=axes[0])
axes[0].set_title("Autocorrelation Function (ACF)")
plot_pacf(df['diff_last_value'].dropna(), ax=axes[1], method='ywm')
axes[1].set_title("Partial Autocorrelation Function (PACF)")
plt.show()

# Fit ARIMA(1,1,0)
model_ar = ARIMA(df['last_value'], order=(1,1,0))
results_ar = model_ar.fit()
print(results_ar.summary())

# Fit ARIMA(0,1,1)
model_ma = ARIMA(df['last_value'], order=(0,1,1))
results_ma = model_ma.fit()
print(results_ma.summary())

# Fit the ARIMA(1,1,0) model for forecasting
model = ARIMA(df['last_value'], order=(1, 1, 0))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Forecast future values
forecast_steps = 30  # Number of days to forecast
forecast = model_fit.forecast(steps=forecast_steps)

# Generate future dates for plotting
future_dates = pd.date_range(start=df.index[-1], periods=forecast_steps+1, freq='D')[1:]

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['last_value'], label="Actual", color='blue')
plt.plot(future_dates, forecast, label="Forecast", color='red')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction Using ARIMA(1,1,0)")
plt.legend()
plt.grid()
plt.show()