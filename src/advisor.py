"""
advisor.py
Takes the user's message + retrieved career context, builds a prompt,
and calls the LLM to generate the career advice response.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load variables from .env
load_dotenv()

# Get Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if API key exists
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Check your .env file."
    )

# Configure Gemini ONCE
genai.configure(api_key=GOOGLE_API_KEY)

# Create Gemini model
_model = genai.GenerativeModel("gemini-3.6-flash")


SYSTEM_PROMPT = """You are a friendly, encouraging career advisor for university students.
You are given some retrieved career profiles (skills, courses, description) that are
relevant to the student's message. Use ONLY this information to ground your advice —
don't invent careers or courses that aren't in the context.

Your reply should:
1. Recommend 1-3 suitable careers from the context, explaining briefly why they fit.
2. Suggest specific courses/certifications from the context for each career.
3. Keep the tone conversational and supportive, not a list dump.
4. If the context doesn't match well, say so honestly and ask a clarifying question.
"""


def build_context(retrieved_careers: list) -> str:
    """Format retrieved career entries into a text block for the prompt."""
    blocks = []

    for c in retrieved_careers:
        blocks.append(
            f"- {c['career']}: {c['description']} "
            f"Skills: {', '.join(c['skills'])}. "
            f"Courses: {', '.join(c['courses'])}."
        )

    return "\n".join(blocks)


def _format_history(chat_history: list) -> str:
    """Turn prior conversation into plain text."""
    lines = []

    for msg in chat_history:
        speaker = "Student" if msg["role"] == "user" else "Advisor"
        lines.append(f"{speaker}: {msg['content']}")

    return "\n".join(lines)


def _call_llm(prompt: str) -> str:
    response = _model.generate_content(prompt)
    return response.text


def get_advice(
    user_message: str,
    retrieved_careers: list,
    chat_history: list = None
) -> str:

    context = build_context(retrieved_careers)

    chat_history = chat_history or []

    history_text = _format_history(chat_history)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Relevant career context:\n{context}\n\n"
        f"Student message: {user_message}"
    )

    return _call_llm(prompt)