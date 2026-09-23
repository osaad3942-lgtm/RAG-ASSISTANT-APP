"""
app.py

Streamlit chat interface for the RAG-Powered Document Assistant.

Ask a question -> the backend retrieves the relevant chunks, calls the LLM,
and returns a grounded answer -> the answer and its cited sources are shown.

Run with:  streamlit run app.py
"""
import re

import streamlit as st

from api_client import APIClientError, ask_question

st.set_page_config(page_title="AI Document Assistant", page_icon="📄")
st.title("AI Document Assistant")

# Chat history lives in session_state so it survives Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message_content(message: dict) -> None:
    """Show one message: the text (or error) and, for answers, the cited sources."""
    if message.get("is_error"):
        st.error(message["content"])
    else:
        st.markdown(message["content"])

    if message.get("sources"):
        st.markdown("**Sources:**")
        for source in message["sources"]:
            st.text(f"📄 {source}")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        render_message_content(message)


# Normal small talk is not a document question, so it gets a short reply right
# here: no backend call and no sources. Anything else goes to the backend.
SMALL_TALK_REPLIES = {
    "Hello!": {
        "hello", "hi", "hey", "hello there", "hi there", "hey there", "hiya",
        "howdy", "yo", "greetings", "good morning", "good afternoon", "good evening",
    },
    "You're welcome!": {
        "thanks", "thank you", "thanks a lot", "thanks so much", "thanks a bunch",
        "thank you so much", "thank you very much", "many thanks", "thx", "ty",
        "appreciate it", "i appreciate it",
    },
    "Goodbye!": {
        "bye", "bye bye", "goodbye", "see you", "see you later", "see ya", "cya",
        "good night", "goodnight", "take care",
    },
    "I'm doing well, thanks for asking!": {
        "how are you", "how are you doing", "how are you today", "how do you do",
        "hows it going", "how is it going", "whats up",
    },
    "Okay!": {
        "ok", "okay", "cool", "great", "nice", "awesome", "alright", "perfect",
        "good", "got it", "sounds good",
    },
    "I'm an AI document assistant. Ask me a question about the lecture material.": {
        "who are you", "what are you", "what can you do", "what do you do",
        "what is your name", "whats your name", "help",
    },
}

_SMALL_TALK_LOOKUP = {
    phrase: reply
    for reply, phrases in SMALL_TALK_REPLIES.items()
    for phrase in phrases
}


def get_small_talk_reply(text: str) -> str | None:
    """Return a short reply if the whole message is small talk (e.g. 'Thanks!'), else None."""
    cleaned = " ".join(re.sub(r"[^a-z\s]", "", text.lower()).split())
    return _SMALL_TALK_LOOKUP.get(cleaned)


def get_assistant_message(question: str) -> dict:
    """Call the backend; on failure return a friendly error instead of a traceback."""
    small_talk_reply = get_small_talk_reply(question)
    if small_talk_reply:
        return {"role": "assistant", "content": small_talk_reply}

    try:
        result = ask_question(question)
        return {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    except APIClientError as exc:
        return {"role": "assistant", "content": str(exc), "is_error": True}


# Replay the conversation so far.
for message in st.session_state.messages:
    render_message(message)

# New question.
question = st.chat_input("Ask a question")

if question:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_message = get_assistant_message(question)
        render_message_content(assistant_message)

    st.session_state.messages.append(assistant_message)
