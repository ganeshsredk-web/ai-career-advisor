"""
app.py
Streamlit chat interface for the AI Career Advisor.
Run with: python -m streamlit run app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
from retriever import CareerRetriever
from advisor import get_advice

st.set_page_config(page_title="AI Career Advisor", page_icon="🎓")
st.title("🎓 AI Career Advisor")
st.caption("Tell me about your skills and interests, and I'll suggest career paths and courses.")


@st.cache_resource
def load_retriever():
    return CareerRetriever(data_path="../data/careers.json")


try:
    with st.spinner("Setting up the advisor (first run may take a moment)..."):
        retriever = load_retriever()
except Exception as e:
    st.error(f"Couldn't load the career database: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Reset button
col1, col2 = st.columns([5, 1])
with col2:
    if st.button("🔄 Reset"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("e.g. I enjoy solving problems and I'm good with data...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                retrieved = retriever.retrieve(user_input, top_k=3)
                reply = get_advice(
                    user_message=user_input,
                    retrieved_careers=retrieved,
                    chat_history=st.session_state.messages[:-1],
                )
            st.markdown(reply)

            with st.expander("📚 Why these careers were suggested"):
                for c in retrieved:
                    st.markdown(f"**{c['career']}** — {c['description']}")
                    st.caption(f"Matched on skills: {', '.join(c['skills'])}")

        except Exception as e:
            reply = f"Something went wrong: {e}. Please try again."
            st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
