import numpy as np


class VectorStore:
    def __init__(self):
        self.vector_data = {}  # A dictionary to store vectors
        self.vector_index = {}  # A dictionary for indexing 
        # structure for retrieval

    def add_vector(self, vector_id, vector):
        """
        Add a vector to the vector store.

        Args:
            vector_id (str or int): A unique id for the vector.
            vector (numpy.array): The vector data to be stored.
        """
        self.vector_data[vector_id] = vector
        self._update_index(vector_id, vector)

    def get_vector(self, vector_id):
        """
        Get a vector from the vector store.

        Args:
            vector_id (str or int): A unique id for the vector.

        Returns:
            numpy.array: The vector data if found, or None if not found.
        """
        return self.vector_data.get(vector_id)

    def _update_index(self, vector_id, vector):
        """
        Update the indexing structure for the vector store.

        Args:
            vector_id (str or int): A unique id for the vector.
            vector (numpy.ndarray): The vector data to be stored.
        """
        for existing_id, existing_vector in self.vector_data.items():
            similarity = np.dot(vector, existing_vector) / \
                (np.linalg.norm(vector) * np.linalg.norm(existing_vector))
            if existing_id not in self.vector_index:
                self.vector_index[existing_id] = {}
            self.vector_index[existing_id][vector_id] = similarity
            self.vector_index[vector_id] = {existing_id: similarity}

    def find_similar_vectors(self, query_vector, num_results=5):
        """
        Find similar vectors to the query vector.

        Args:
            query_vector (numpy.ndarray): The query vector.
            num_results (int): The number of results to return.

        Returns:
            list: A list of tuples of the form (vector_id, similarity_score).
        """
        results = []
        for vector_id, vector in self.vector_data.items():
            similarity = np.dot(query_vector, vector) / \
                (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            results.append((vector_id, similarity))

        # Sort the results by similarity score in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        # Return the top N results
        return results[:num_results]
