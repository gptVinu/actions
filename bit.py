plays = {
    "Anthony and Cleopatra": "Anthony is there, Brutus is Caeser is with Cleopatra mercy worser.",
    "Julius Caeser": "Anthony is there, Brutus is Caeser is but Calpurnia is.",
    "The Tempest": "mercy worser",
    "Hamlet": "Caeser and Brutus are present with mercy and worser",
    "Othello": "Caeser is present with mercy",
    "Macbeth": "Anthony is there; Caeser, mercy.",
}

words = ["Anthony", "Brutus", "Caeser", "Calpurnia", "Cleopatra", "mercy", "worser"]

# Create the matrix
matrix = [[1 if word in play else 0 for play in plays.values()] for word in words]

# Print the matrix with headers
print(f"{'Words/Plays':<12}", end="\t")
for play in plays.keys():
    print(f"{play:<20}", end="\t")
print()

for i, word in enumerate(words):
    print(f"{word:<12}", end="\t")
    for value in matrix[i]:
        print(f"{value:<20}", end="\t")
    print()

print("-" * 120)

# Convert binary presence to integers
vector_dict = {word: int("".join(map(str, row)), 2) for word, row in zip(words, matrix)}

print(f"Binary presence as integers:\n\t{vector_dict}")
print("-" * 120)

# Get condition from user
condition = input("Enter your condition: ")
processed_condition = condition

# Replace words with their binary integer representation
for word in words:
    if word in processed_condition:
        processed_condition = processed_condition.replace(word, str(vector_dict[word]))

print(f"\t--> Binary corresponding integer representation: {processed_condition}")
print("-" * 120)

# Replace logical operators for evaluation
processed_condition = processed_condition.replace("not", "and~").replace("or", "|").replace("and", "&")

# Evaluate the condition with error handling
try:
    binary_result = bin(eval(processed_condition)).replace("0b", "")
    print("Binary string form:", binary_result)
    print("-" * 120)

    print(f"The plays which satisfy the condition '{condition}' are:")
    for i, bit in enumerate(binary_result):
        if bit == "1":
            print(f"\t=> {list(plays.keys())[i]}")

except NameError as e:
    print(f"Error: {e}. Please ensure your condition only uses defined words: {', '.join(words)}.")
except Exception as e:
    print(f"An error occurred: {e}")