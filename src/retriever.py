"""
retriever.py
Loads the career dataset, embeds it using Gemini's embedding API, and
retrieves the most relevant career entries for a given user query.

Embeddings are cached to disk (embeddings_cache.json) so the app doesn't
re-call the embedding API on every startup — only when careers.json changes.
This also avoids needing sentence-transformers/torch, which is heavy and
was causing install issues on Windows.
"""

import json
import os
import hashlib
import numpy as np
import faiss
import google.generativeai as genai

DATA_PATH = "../data/careers.json"
CACHE_PATH = "../data/embeddings_cache.json"
EMBED_MODEL = "models/gemini-embedding-001"

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


class CareerRetriever:
    def __init__(self, data_path: str = DATA_PATH, cache_path: str = CACHE_PATH):
        self.data_path = data_path
        self.cache_path = cache_path
        self.careers = self._load_data()
        self.embeddings = self._load_or_build_embeddings()
        self.index = self._build_index(self.embeddings)

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

    def _data_hash(self) -> str:
        """Hash of the raw dataset file — used to detect if careers.json changed."""
        with open(self.data_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _load_or_build_embeddings(self) -> np.ndarray:
        current_hash = self._data_hash()

        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                cache = json.load(f)
            if cache.get("hash") == current_hash:
                return np.array(cache["embeddings"], dtype="float32")

        # No valid cache — embed via Gemini API and save
        texts = [self._entry_to_text(c) for c in self.careers]
        embeddings = []
        for text in texts:
            result = genai.embed_content(model=EMBED_MODEL, content=text)
            embeddings.append(result["embedding"])
        embeddings = np.array(embeddings, dtype="float32")

        with open(self.cache_path, "w") as f:
            json.dump({"hash": current_hash, "embeddings": embeddings.tolist()}, f)

        return embeddings

    def _build_index(self, embeddings: np.ndarray):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def _embed_query(self, query: str) -> np.ndarray:
        result = genai.embed_content(model=EMBED_MODEL, content=query)
        return np.array([result["embedding"]], dtype="float32")

    def retrieve(self, query: str, top_k: int = 3):
        """Return the top_k most relevant career entries for a query."""
        query_vec = self._embed_query(query)
        distances, indices = self.index.search(query_vec, top_k)
        return [self.careers[i] for i in indices[0]]


if __name__ == "__main__":
    # quick manual test
    retriever = CareerRetriever()
    results = retriever.retrieve("I like coding in Python and building models")
    for r in results:
        print(r["career"])
