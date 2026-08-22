"""
advisor.py
Takes the user's message + retrieved career context, builds a prompt,
and calls the LLM to generate the career advice response.

Uses Google Gemini's API. To switch to OpenAI or Claude, only
_call_llm() needs to change — everything else stays the same.
"""

import os
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-flash-lite-latest")

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
    """Turn prior turns into plain text (Gemini has no 'system' role, so we
    fold everything into one prompt for simplicity)."""
    lines = []
    for msg in chat_history:
        speaker = "Student" if msg["role"] == "user" else "Advisor"
        lines.append(f"{speaker}: {msg['content']}")
    return "\n".join(lines)


def _call_llm(prompt: str) -> str:
    try:
        response = _model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Friendly fallback instead of crashing the whole app
        return (
            "Sorry, I'm having trouble reaching the AI service right now "
            f"({type(e).__name__}). Please try again in a moment."
        )


MAX_HISTORY_TURNS = 5  # keep prompts from growing too large in long chats


def get_advice(user_message: str, retrieved_careers: list, chat_history: list = None) -> str:
    """
    user_message: the latest thing the student typed
    retrieved_careers: list of career dicts from CareerRetriever.retrieve()
    chat_history: list of {"role": "user"/"assistant", "content": ...} from earlier turns
    """
    context = build_context(retrieved_careers)
    chat_history = (chat_history or [])[-MAX_HISTORY_TURNS * 2:]
    history_text = _format_history(chat_history)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Relevant career context:\n{context}\n\n"
        f"Student message: {user_message}"
    )

    return _call_llm(prompt)
