import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc

file_path = "C:/Users/vinyak/Downloads/myData.csv"
df = pd.read_csv(file_path)

df['Outcome'] = df['Outcome'].apply(lambda x: 1 if x >= 0.5 else 0)
X, y = df.drop('Outcome', axis=1), df['Outcome']

X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

nb = GaussianNB().fit(X_train, y_train)
y_pred, y_prob = nb.predict(X_test), nb.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
prec = precision_score(y_test, y_pred)
sens = recall_score(y_test, y_pred)
spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
se = np.sqrt((acc * (1 - acc)) / len(y_test))

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Print Metrics
print("Naive Bayes Model Metrics:")
print(f"  Accuracy: {acc:.3f}")
print(f"  Precision: {prec:.3f}")
print(f"  Sensitivity (Recall): {sens:.3f}")
print(f"  Specificity: {spec:.3f}")
print(f"  AUC: {roc_auc:.3f}")
print(f"  Standard Error: {se:.3f}")
print("\nConfusion Matrix:")
print(cm)
print("\nPredictions:", y_pred)