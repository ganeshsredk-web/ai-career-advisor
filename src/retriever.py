"""
retriever.py
Loads the career dataset, embeds it, and retrieves the most relevant
career entries for a given user query (skills/interests).
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

DATA_PATH = "../data/careers.json"
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for this project


class CareerRetriever:
    def __init__(self, data_path: str = DATA_PATH, model_name: str = MODEL_NAME):
        self.data_path = data_path
        self.model = SentenceTransformer(model_name)
        self.careers = self._load_data()
        self.index, self.embeddings = self._build_index()

    def _load_data(self):
        with open(self.data_path, "r") as f:
            return json.load(f)

    def _entry_to_text(self, entry: dict) -> str:
        """Turn one career record into a single text blob for embedding."""
        return (
            f"{entry['career']}. {entry['description']} "
            f"Skills: {', '.join(entry['skills'])}. "
            f"Courses: {', '.join(entry['courses'])}."
        )

    def _build_index(self):
        texts = [self._entry_to_text(c) for c in self.careers]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index, embeddings

    def retrieve(self, query: str, top_k: int = 3):
        """Return the top_k most relevant career entries for a query."""
        query_vec = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k)
        return [self.careers[i] for i in indices[0]]


if __name__ == "__main__":
    # quick manual test
    retriever = CareerRetriever()
    results = retriever.retrieve("I like coding in Python and building models")
    for r in results:
        print(r["career"])
