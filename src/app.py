import streamlit as st
from vector_db import load_faiss, load_chroma
from groq import Groq

# 🔑 API
import os

api_key = os.getenv("GROQ_API_KEY") # 🔥 apni key daal

# load DBs
@st.cache_resource
def load_dbs():
    return load_faiss(), load_chroma()

faiss_db, chroma_db = load_dbs()

# UI
st.title("⚡ Smart AI RAG Chatbot")

# session
if "messages" not in st.session_state:
    st.session_state.messages = []

# answer function
def get_answer(db, model, query):
    results = db.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in results])

    prompt = f"""
Answer clearly and shortly.

Context:
{context}

Question: {query}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


# input
with st.form("chat_form", clear_on_submit=True):
    query = st.text_input("💬 Ask anything...")
    submitted = st.form_submit_button("🚀 Send")

if submitted and query:
    st.session_state.messages.append(("user", query))

    try:
        ans1 = get_answer(faiss_db, "llama-3.1-8b-instant", query)
        ans2 = get_answer(chroma_db, "llama-3.1-8b-instant", query) 
        final_answer = f"""
🔹 FAISS + MiniLM + LLaMA:
{ans1}

🔹 Chroma + MPNet + LLaMA:
{ans2}
"""

    except Exception as e:
        final_answer = f"❌ ERROR: {str(e)}"

    st.session_state.messages.append(("bot", final_answer))

# display
for role, msg in st.session_state.messages:
    if role == "user":
        st.write(f"🧑 {msg}")
    else:
        st.write(f"🤖 {msg}")