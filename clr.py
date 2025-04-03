import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

file_path = "C:/Users/vinyak/Downloads/myData.csv"
df = pd.read_csv(file_path)

X, y = df[['Insulin']], df['Outcome']
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred_binary)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Values'), plt.ylabel('Predicted Values'), plt.title('Actual vs Predicted - Linear Regression'), plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='green', alpha=0.6, edgecolor='k')
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.xlabel('Predicted Values'), plt.ylabel('Residuals'), plt.title('Residuals - Linear Regression'), plt.show()

print(f"Mean Squared Error: {mse:.3f}\nR-squared: {r2:.3f}\nAccuracy: {acc:.3f}")