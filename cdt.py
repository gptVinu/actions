import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc

file_path = "C:/Users/vinyak/Downloads/myData.csv"
df = pd.read_csv(file_path)

df['Outcome'] = df['Outcome'].apply(lambda x: 1 if x >= 0.5 else 0)
X, y = df.drop('Outcome', axis=1), df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
y_pred, y_prob = clf.predict(X_test), clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
prec = precision_score(y_test, y_pred)
sens = recall_score(y_test, y_pred)
spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
se = np.sqrt((acc * (1 - acc)) / len(y_test))

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('FPR'), plt.ylabel('TPR'), plt.title('Decision Tree ROC'), plt.legend(loc="lower right"), plt.show()

plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=['0', '1'], filled=True, max_depth=3)
plt.title("Decision Tree (Max Depth = 3)"), plt.show()

# Output
print("Decision Tree Metrics:")
print(f"  Accuracy: {acc:.3f}, Precision: {prec:.3f}, Sensitivity: {sens:.3f}")
print(f"  Specificity: {spec:.3f}, AUC: {roc_auc:.3f}, Std Err: {se:.3f}")
print("\nConfusion Matrix:\n", cm)
print("\nPredictions:", y_pred)