import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.utils import resample

file_path = "C:/Users/vinyak/Downloads/myData.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    x, y = df[['Age', 'Glucose', 'SkinThickness']].values, df["Outcome"].values  # Feature selection

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)  # Train-test split

    # Scaling
    scaler = StandardScaler() #create one instance of standard scaler
    x_train_scaled = scaler.fit_transform(x_train) #fit and transform the training data
    x_test_scaled = scaler.transform(x_test) #transform the test data, using the same scaler

    dt_classifier = DecisionTreeClassifier(random_state=0)
    classifier = AdaBoostClassifier(estimator=dt_classifier, n_estimators=50, random_state=42).fit(x_train_scaled, y_train) #Boosting
    y_pred = classifier.predict(x_test_scaled)
    y_pred_prob = classifier.predict_proba(x_test_scaled)[:, 1]

    def plot_metrics(y_true, y_pred, y_prob):
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
        print(f"Precision: {precision_score(y_true, y_pred):.2f}")
        print(f"Recall: {recall_score(y_true, y_pred):.2f}")

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate'), plt.title('ROC Curve'), plt.legend(loc="lower right"), plt.show()

    plot_metrics(y_test, y_pred, y_pred_prob)

    plt.figure(figsize=(12, 8))
    plot_tree(dt_classifier.fit(x_train_scaled, y_train), feature_names=['Age', 'Glucose', 'SkinThickness'], class_names=['No Diabetes', 'Diabetes'], filled=True, max_depth=3)
    plt.title("Decision Tree (Max Depth = 3)"), plt.show()

    # Cross-validation
    cv_scores = cross_val_score(dt_classifier, x_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f'Cross-Validation Results (Accuracy): {cv_scores}\nMean Accuracy: {cv_scores.mean()}')

    # Bootstrap
    x_train_boot, y_train_boot = resample(x_train_scaled, y_train, replace=True, n_samples=len(x_train_scaled), random_state=42)
    dt_classifier.fit(x_train_boot, y_train_boot)
    y_pred_boot = dt_classifier.predict(x_test_scaled)
    print(f"Bootstrap Accuracy: {accuracy_score(y_test, y_pred_boot):.2f}")

    # Bagging
    bagging_classifier = BaggingClassifier(estimator=dt_classifier, n_estimators=50, random_state=0).fit(x_train_scaled, y_train)
    y_pred_bagging = bagging_classifier.predict(x_test_scaled)
    print(f"Bagging Accuracy: {accuracy_score(y_test, y_pred_bagging):.2f}")

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap of Diabetes Dataset"), plt.show()

else:
    print(f"File not found at {file_path}")