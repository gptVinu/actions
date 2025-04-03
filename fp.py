import pyfpgrowth
import csv

# Example transactions
transactions = [[1, 2, 5], [2, 4], [2, 3], [1, 2, 4], [1, 3], [2, 3], [1, 3], [1, 2, 3, 5], [1, 2, 3]]

# Load and preprocess groceries data
with open('groceries.csv', encoding="utf8", newline='') as f:
    data = [list(filter(None, row)) for row in csv.reader(f)][1:]  # Skip header, remove empty strings

print("Sample Groceries Data:\n", data[:10])

# Find frequent patterns
patterns = pyfpgrowth.find_frequent_patterns(data, 2)
print("\nFrequent Patterns:\n", patterns)

# Generate association rules
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
print("\nAssociation Rules:\n", rules)