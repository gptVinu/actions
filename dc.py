import pandas as pd

# Sample data with missing values and inconsistencies
data = {
    'Name': ['Alice', 'Bob', None, 'Dave'],
    'Age': [25, None, 30, 35],
    'Salary': [50000, 60000, None, 80000],
    'City': ['New York', 'Los Angeles', 'New York', 'San Francisco']
}

# Create DataFrame
df = pd.DataFrame(data)

# Handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].median(), inplace=True)
df['Name'].fillna('Unknown', inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Standardize text data
df['City'] = df['City'].str.lower()

# Remove outliers (e.g., Salary > 100000)
df = df[df['Salary'] <= 100000]

# Final cleaned DataFrame
print("Cleaned DataFrame:\n", df)