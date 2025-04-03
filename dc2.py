import pandas as pd

file_path = "C:/Users/vinyak/Downloads/myData.csv"
df = pd.read_csv(file_path)

print("Original Data (First 3 Rows):\n", df.head(3), "\n")
print("Column Data Types:\n", df.dtypes, "\n")

# Check for NaN values in any column
invalid_rows = df[df.isna().any(axis=1)]
print("Rows with NaN values:\n", invalid_rows, "\n")

# Drop rows with NaN values
df = df.dropna()
print("Rows with NaN values after drop:\n", df[df.isna().any(axis=1)], "\n")

# Convert 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction' to numeric, handling errors
numeric_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Re-check for NaN values after numeric conversion
invalid_rows_after_conversion = df[df.isna().any(axis=1)]
print("Rows with NaN values after numeric conversion:\n", invalid_rows_after_conversion, "\n")

# Drop any newly introduced NaN values
df = df.dropna()

# Round numerical columns to 2 decimal places
for col in numeric_cols:
    df[col] = df[col].round(2)

print("Data Types After Transformation:\n", df.dtypes, "\n")
print("Transformed DataFrame (First 4 Rows):\n", df.head(4))