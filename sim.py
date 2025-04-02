import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")


def process_text(file_path):
    with open(file_path, "r") as file:
        raw_text = file.read()
    tokens = word_tokenize(raw_text)
    words = [word.lower() for word in tokens]
    stemmer = nltk.PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in stemmed_words if word not in stop_words]
    word_counts = nltk.defaultdict(int)
    for word in filtered_words:
        word_counts[word] += 1
    return word_counts


def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def calculate_similarity(word_counts1, word_counts2):
    all_words = list(set(word_counts1.keys()).union(set(word_counts2.keys())))
    vector1 = np.array([word_counts1.get(word, 0) for word in all_words])
    vector2 = np.array([word_counts2.get(word, 0) for word in all_words])
    return cosine_similarity(vector1, vector2)


if __name__ == "__main__":
    file1_path = "C:/Users/vinyak/OneDrive/Documents/Vinayak_Official/SEM-VI/IR/myData1.txt"
    file2_path = "C:/Users/vinyak/OneDrive/Documents/Vinayak_Official/SEM-VI/IR/myData2.txt"
    word_counts1 = process_text(file1_path)
    word_counts2 = process_text(file2_path)
    similarity_score = calculate_similarity(word_counts1, word_counts2)
    print("Similarity between the two text documents:", similarity_score)