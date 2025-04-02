import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def filter_stopwords(text_or_file, is_file=False):
    """Filters stop words from a given text or file."""
    if is_file:
        try:
            with open(text_or_file, 'r') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: File '{text_or_file}' not found.")
            return
    else:
        text = text_or_file

    word_tokens = word_tokenize(text)
    filtered_words = [word for word in word_tokens if word.lower() not in stop_words]

    if is_file:
        output_filename = "filtered_output.txt"
        with open(output_filename, 'w') as write_file:
            write_file.write(' '.join(filtered_words))
        print(f"Filtered text written to '{output_filename}'.")
    else:
        print("Filtered Sentence:", ' '.join(filtered_words))

if __name__ == "__main__":
    while True:
        print("\nChoose an option:")
        print("1. Filter stop words from a sentence.")
        print("2. Filter stop words from a file.")
        print("3. Exit.")

        choice = input("Enter your choice: ")

        if choice == "1":
            sentence = input("Enter a sentence: ")
            filter_stopwords(sentence)
        elif choice == "2":
            filename = input("Enter the filename: ")
            filter_stopwords(filename, is_file=True)
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")

    print("Exiting...")