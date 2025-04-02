import numpy as np

DAMPING_FACTOR = 0.85

def calculate_page_rank(graph, damping_factor, iterations):
    page_names = list(graph.keys())
    num_pages = len(page_names)
    adjacency_matrix = np.zeros((num_pages, num_pages))

    # Create adjacency matrix
    for i, page in enumerate(page_names):
        outbound_links = graph[page]
        if outbound_links:
            for link in outbound_links:
                j = page_names.index(link)
                adjacency_matrix[j, i] = 1 / len(outbound_links)

    page_rank = np.ones(num_pages) / num_pages

    # Perform PageRank iterations
    for _ in range(iterations):
        page_rank = (1 - damping_factor) / num_pages + damping_factor * np.dot(adjacency_matrix, page_rank)

    # Create PageRank dictionary
    page_rank_dict = {page_names[i]: page_rank[i] for i in range(num_pages)}

    return page_rank_dict

# Example graph
graph = {
    'A': ['B', 'C'],
    'B': ['C'],
    'C': ['A']
}

# Calculate PageRank scores
page_rank_scores = calculate_page_rank(graph, DAMPING_FACTOR, 3)

# Print PageRank scores
print("PageRank scores:")
for page, score in page_rank_scores.items():
    print(f"\tPage-{page}: {score:.6f}") 