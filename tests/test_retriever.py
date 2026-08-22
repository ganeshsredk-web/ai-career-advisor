import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from retriever import CareerRetriever


def test_retrieve_returns_results():
    retriever = CareerRetriever(data_path="../data/careers.json")
    results = retriever.retrieve("I like coding and machine learning", top_k=3)
    assert len(results) == 3
    assert all("career" in r for r in results)


def test_retrieve_relevant_match():
    retriever = CareerRetriever(data_path="../data/careers.json")
    results = retriever.retrieve("I want to design app interfaces", top_k=1)
    assert results[0]["career"] in ["UX/UI Designer", "Software Developer"]


if __name__ == "__main__":
    test_retrieve_returns_results()
    test_retrieve_relevant_match()
    print("All tests passed.")
