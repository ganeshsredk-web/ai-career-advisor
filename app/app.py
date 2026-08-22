"""
app.py
Streamlit chat interface for the AI Career Advisor.
Run with: python -m streamlit run app/app.py
"""

import sys
import os

# Add src folder to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

sys.path.append(SRC_DIR)

import streamlit as st
from retriever import CareerRetriever
from advisor import get_advice


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="🎓"
)

st.title("🎓 AI Career Advisor")
st.caption(
    "Tell me about your skills and interests, "
    "and I'll suggest career paths and courses."
)


# --------------------------------------------------
# Career data path
# --------------------------------------------------

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "careers.json"
)


# --------------------------------------------------
# Load career retriever
# --------------------------------------------------

@st.cache_resource
def load_retriever():
    return CareerRetriever(data_path=DATA_PATH)


retriever = load_retriever()


# --------------------------------------------------
# Chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------

user_input = st.chat_input(
    "e.g. I enjoy solving problems and I'm good with data..."
)


if user_input:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)


    # --------------------------------------------------
    # Retrieve relevant careers
    # --------------------------------------------------

    retrieved = retriever.retrieve(
        user_input,
        top_k=3
    )


    # --------------------------------------------------
    # Generate AI response
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            reply = get_advice(
                user_message=user_input,
                retrieved_careers=retrieved,
                chat_history=st.session_state.messages[:-1]
            )

            st.markdown(reply)


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply
        }
    )


    # --------------------------------------------------
    # Show retrieved careers
    # --------------------------------------------------

    with st.expander("📚 Careers retrieved for this message"):

        for c in retrieved:

            st.write(
                f"**{c['career']}** — {c['description']}"
            )