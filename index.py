import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    return [word for word in text if word not in stop_words]

def build_inverted_index(documents):
    index = {}
    for doc_id, content in enumerate(documents):
        for word in preprocess_text(content):
            index.setdefault(word, []).append(doc_id)
    return index

def search(index, query):
    query_terms = preprocess_text(query)
    results = set()
    if not query_terms:
      return results
    first_term_results = index.get(query_terms[0], [])
    results.update(first_term_results)

    for term in query_terms[1:]:
        results.intersection_update(index.get(term, []))

    return results

if __name__ == "__main__":
    documents = [
        "The quick brown fox jumped over the lazy dog",
        "The lazy dog slept in the sun",
        "Information retrieval is an essential aspect of search engines.",
        "The field of information retrieval focuses on algorithms.",
        "Search engines use retrieval techniques to improve performance.",
        "Deep learning models are used for information retrieval tasks."
    ]

    index = build_inverted_index(documents)
    query = "information retrieval"
    results = search(index, query)

    print("Inverted Index:")
    for word, doc_ids in index.items():
        print(f"{word}: {doc_ids}")

    print(f"\nDocuments containing the query '{query}': {sorted(results)}")