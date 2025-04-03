import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file_path = "C:/Users/vinyak/Downloads/myData.csv"
df = pd.read_csv(file_path)

df['Outcome'] = df['Outcome'].apply(lambda x: 1 if x >= 0.5 else 0)
X, y = df.drop('Outcome', axis=1), df['Outcome']

X_scaled = StandardScaler().fit_transform(X)

n_components = 7
pca = PCA(n_components=n_components).fit(X_scaled)
X_pca = pca.transform(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
total_variance_explained = np.sum(explained_variance_ratio)

print("Explained Variance Ratio:", explained_variance_ratio)
print("Total Variance Explained:", total_variance_explained)

plt.figure(figsize=(8, 5))
plt.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.7, align='center', label='Individual Explained Variance')
plt.step(range(1, n_components + 1), np.cumsum(explained_variance_ratio), where='mid', label='Cumulative Explained Variance')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.legend(loc='best')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)