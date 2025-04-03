import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc

file_path = "C:/Users/vinyak/Downloads/myData.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    x, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    classifier = DecisionTreeClassifier(random_state=0).fit(x_train, y_train)
    y_pred, y_pred_prob = classifier.predict(x_test), classifier.predict_proba(x_test)[:, 1]

    def plot_metrics(y_true, y_pred, y_prob):
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
        print(f"Precision: {precision_score(y_true, y_pred):.2f}")
        print(f"Recall: {recall_score(y_true, y_pred):.2f}")

        fpr, tpr, _ = roc_curve(y_true, y_prob) # Only call roc_curve once.
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'), plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate'), plt.title('ROC Curve'), plt.legend(loc="lower right"), plt.show()

    plot_metrics(y_test, y_pred, y_pred_prob)

    plt.figure(figsize=(12, 8))
    plot_tree(classifier, feature_names=df.columns[:-1], class_names=['No Diabetes', 'Diabetes'], filled=True, max_depth=3)
    plt.title("Decision Tree (Max Depth = 3)"), plt.show()
else:
    print(f"File not found at {file_path}")