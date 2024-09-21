from vector_store import VectorStore
import numpy as np

# Create a vector store instance
vector_store = VectorStore()

# Define your sentences
sentences = [
    "I eat mango",
    "mango is my favorite fruit",
    "mango, apple, and oranges are fruits",
    "fruits are good for health"
]

# Tokenization and vocabulary building
vocabulary = set()

for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)

# Assign unique indices to words in the vocab
word_to_index = {word: i for i, word in enumerate(vocabulary)}

# Initialize a dictionary to store sentence vectors
sentence_vectors = {}

# Vectorization
for sentence in sentences:
    tokens = sentence.lower().split()
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        vector[word_to_index[token]] += 1
    # Store the vector for the current sentence
    sentence_vectors[sentence] = vector

# Add vectors to the vector store
for sentence, vector in sentence_vectors.items():
    vector_store.add_vector(sentence, vector)

# Searching for the similarity
query_sentence = "Mango is the best fruit"
query_vector = np.zeros(len(vocabulary))
query_tokens = query_sentence.lower().split()
for token in query_tokens:
    if token in word_to_index:  # Check if token exists in the vocabulary
        query_vector[word_to_index[token]] += 1

# Find similar sentences
similar_sentences = vector_store.find_similar_vectors(
    query_vector, num_results=2)

# Output results
print("Query sentence:", query_sentence)
print("Similar sentences:")
for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity = {similarity:.4f}")
