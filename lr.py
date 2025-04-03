import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv("C:\\Users\\vinyak\\Downloads\\myData.csv")

# Prepare data
X = data['Age'].values.reshape(-1, 1)
y = data['Glucose']

# Train model
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

# Calculate metrics
correlation = np.corrcoef(data['Age'], data['Glucose'])[0, 1]
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Evaluate model fit
fit_strength = "strong" if r2 >= 0.7 else "moderate" if 0.5 <= r2 < 0.7 else "weak"

# Display results
print(f"Correlation: {correlation:.3f},\nRMSE: {rmse:.3f},\nR-squared: {r2:.3f}")
print(f"Model fit: {fit_strength}.")

# Plotting
plt.figure(figsize=(12, 6))
plt.scatter(data['Age'], y, alpha=0.5, label='Actual Data')
plt.plot(np.sort(X.flatten()), y_pred[np.argsort(X.flatten())], color='red', label='Regression Line')
plt.title('Linear Regression: Age vs Glucose')
plt.xlabel('Age')
plt.ylabel('Glucose')
plt.legend()
plt.grid(alpha=0.3)
plt.show()