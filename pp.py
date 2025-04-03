import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

file_path = "C:/Users/vinyak/Downloads/myData.csv"
df = pd.read_csv(file_path)

# Correlation Heatmap
corr_matrix = df[['Age', 'Glucose', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Outcome']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap (Diabetes Dataset)")
plt.show()

# Chi-Square Test (Outcome vs. Glucose Categories)
glucose_bins = pd.cut(df['Glucose'], bins=[0, 100, 150, 200], labels=['Low', 'Medium', 'High'])
contingency_table = pd.crosstab(glucose_bins, df['Outcome'])

chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print("\nChi-Square Test Results (Glucose vs. Outcome)")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies:\n{expected}")