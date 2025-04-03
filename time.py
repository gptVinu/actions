import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error #scikit-learn
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv"
data = pd.read_csv(url, header=0, parse_dates=[0], index_col=0)

# Split data into training and testing sets
train = data.iloc[:-12]
test = data.iloc[-12:]

# ARIMA Model
arima_model = ARIMA(train, order=(5, 1, 0))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=12)

# SARIMAX Model
sarimax_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_result = sarimax_model.fit()
sarimax_forecast = sarimax_result.forecast(steps=12)

# Plotting the forecasts
plt.figure(figsize=(12, 6))
plt.plot(train, label='Train', color='blue')
plt.plot(test, label='Test', color='green')
plt.plot(arima_forecast, label='ARIMA Forecast', color='red', linestyle='--')
plt.plot(sarimax_forecast, label='SARIMAX Forecast', color='orange', linestyle='--')
plt.legend()
plt.title('ARIMA and SARIMAX Forecasting of Air Passengers')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.grid(True)
plt.tight_layout() # Improves plot layout
plt.show()

# Evaluate the models
def evaluate_forecast(actual, predicted, model_name):
    """
    Evaluates the forecast using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

    Args:
        actual (pd.Series or np.array): Actual values.
        predicted (pd.Series or np.array): Predicted values.
        model_name (str): Name of the model for reporting.

    Returns:
        tuple: MAE and RMSE values.
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"{model_name} MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, rmse

# Calculate and print metrics
print("Evaluation Metrics:")
arima_mae, arima_rmse = evaluate_forecast(test, arima_forecast, "ARIMA")
sarimax_mae, sarimax_rmse = evaluate_forecast(test, sarimax_forecast, "SARIMAX")