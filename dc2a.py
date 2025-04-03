import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load dataset
file_path = "C:/Users/vinyak/Downloads/myData.csv"
df = pd.read_csv(file_path)
print("Original Data:\n", df.head(3), "\n")

# Select numeric columns for scaling
numeric_cols = df.select_dtypes(include=['float64', 'int64']).drop('Outcome', axis=1).columns # Exclude 'Outcome'

# Min-Max Scaling
min_max_scaler = MinMaxScaler()
df_min_max_scaled = df.copy()
df_min_max_scaled[numeric_cols] = min_max_scaler.fit_transform(df[numeric_cols])

# Z-score Normalization (Standard Scaling)
standard_scaler = StandardScaler()
df_zscore_normalized = df.copy()
df_zscore_normalized[numeric_cols] = standard_scaler.fit_transform(df[numeric_cols])

# Output results
print("Min-Max Scaled Data:\n", df_min_max_scaled.head(3), "\n")
print("Z-Score Normalized Data:\n", df_zscore_normalized.head(3))