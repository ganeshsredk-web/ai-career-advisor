# AI Career Advisor (RAG-based)

A conversational AI career advisor for university students, built for BIT4543
Artificial Intelligence group project.

## How it works
1. **Retrieval** — `src/retriever.py` embeds a small dataset of career profiles
   (`data/careers.json`) and retrieves the most relevant ones for a user's message
   using sentence embeddings + FAISS similarity search.
2. **Generation** — `src/advisor.py` takes the retrieved career context + the
   user's message, builds a prompt, and calls an LLM (Google Gemini) to
   generate a natural-language recommendation.
3. **Interface** — `app/app.py` is a Streamlit chat app tying both together.

## Setup
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
```

## Run
```bash
cd app
streamlit run app.py
```

## Project structure
```
data/       career dataset (JSON)
src/        retrieval + LLM logic
app/        Streamlit chat frontend
tests/      basic unit tests for retrieval
models/     (reserved — not needed unless you add a trained classifier)
notebooks/  (for experimentation / prototyping)
docs/       (for report, diagrams, etc. — added later)
results/    (evaluation outputs, screenshots, etc.)
```

## Swapping the LLM provider
Only `_call_llm()` in `src/advisor.py` needs to change to use OpenAI or Claude
instead of Gemini — the rest of the pipeline (retrieval, prompt building,
chat history) stays the same.

## Team task split (suggested)
- Data collection & dataset curation (`data/careers.json`)
- Retrieval/embedding logic (`src/retriever.py`)
- LLM prompting & advisor logic (`src/advisor.py`)
- Frontend (`app/app.py`)
- Testing & documentation (`tests/`, `docs/`)
