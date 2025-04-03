import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load data
data = pd.read_csv("groceries.csv")

# Preprocess transactions
transactions = [[item for item in row if item != 'nan'] for row in data.values.astype(str).tolist()]
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

# Frequent itemsets
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

# Plot top 15 frequent itemsets
plt.figure(figsize=(12, 6))
sns.barplot(x='itemsets', y='support', data=frequent_itemsets.nlargest(15, 'support'), palette=['#FFB6C1', '#ADD8E6', '#98FB98', '#FFD700', '#FFA07A', '#87CEEB', '#FFC0CB', '#FF69B4', '#00FA9A', '#FF6347'])
plt.xticks(rotation=90)
plt.show()

# Association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1, num_itemsets=4)

# Rule filtering and display
rules["antecedent_len"] = rules["antecedents"].apply(len)
rules["consequent_len"] = rules["consequents"].apply(len)

filtered_rules = rules[
    (rules["antecedent_len"] >= 2) &
    (rules["confidence"] > 0.3) &
    (rules["lift"] > 1)
].sort_values(by=["lift", "support"], ascending=False)
print("\nFiltered Rules (antecedents >= 2, confidence > 0.3, lift > 1):\n", filtered_rules)

filtered_rules2 = rules[
    (rules["consequent_len"] >= 2) &
    (rules["lift"] > 1)
].sort_values(by=["lift", "confidence"], ascending=False)
print("\nFiltered Rules (consequents >= 2, lift > 1):\n", filtered_rules2)

filtered_rules2["lift"] = filtered_rules2["support"] / (filtered_rules2["antecedent_len"] * filtered_rules2["consequent_len"])
print(filtered_rules2)

# Accuracy metrics (all rules, no filtering)
print("\n---------------------ACCURACY METRICS---------------------------")
print("\nLift:\n", association_rules(frequent_itemsets, metric="lift", min_threshold=1, num_itemsets=4))
print("\nConfidence:\n", association_rules(frequent_itemsets, metric="confidence", min_threshold=1, num_itemsets=4))
print("\nLeverage:\n", association_rules(frequent_itemsets, metric="leverage", min_threshold=1, num_itemsets=4))
print("\nConviction:\n", association_rules(frequent_itemsets, metric="conviction", min_threshold=1, num_itemsets=4))