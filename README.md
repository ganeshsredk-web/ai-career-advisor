# AI Career Advisor (RAG-based)

A conversational AI career advisor for university students, built for BIT4543
Artificial Intelligence group project.

## How it works
1. **Retrieval** — `src/retriever.py` embeds a small dataset of career profiles
   (`data/careers.json`) using Gemini's embedding API and retrieves the most
   relevant ones for a user's message via FAISS similarity search. Embeddings
   are cached to `data/embeddings_cache.json` so they're only recomputed when
   the dataset changes — this keeps startup fast.
2. **Generation** — `src/advisor.py` takes the retrieved career context + the
   user's message, builds a prompt, and calls Gemini to generate a
   natural-language recommendation. API errors are handled gracefully instead
   of crashing the app.
3. **Interface** — `app/app.py` is a Streamlit chat app tying both together,
   with a reset button and a "why these careers were suggested" breakdown.

## Setup
```bash
pip install -r requirements.txt
```
Set your Gemini API key for the current terminal session:
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"
```
```bash
# Mac/Linux
export GEMINI_API_KEY="your-key-here"
```

## Run
```bash
cd app
python -m streamlit run app.py
```

## Project structure
```
data/       career dataset (JSON) + cached embeddings
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

## Deploying for the demo (Streamlit Community Cloud)
1. Push this repo to GitHub (already done).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click "New app", select this repo, set the main file path to `app/app.py`.
4. Under "Advanced settings" → "Secrets", add:
   ```
   GEMINI_API_KEY = "your-key-here"
   ```
5. Deploy — you'll get a public link to share with your team/lecturer.

## Team task split (suggested)
- Data collection & dataset curation (`data/careers.json`)
- Retrieval/embedding logic (`src/retriever.py`)
- LLM prompting & advisor logic (`src/advisor.py`)
- Frontend (`app/app.py`)
- Testing & documentation (`tests/`, `docs/`)
