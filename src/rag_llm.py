from langchain_community.vectorstores import FAISS, Chroma
from embedding import create_embeddings
from groq import Groq

# 🔑 API
import os

api_key = os.getenv("GROQ_API_KEY")

# 🔹 load both DBs
def load_dbs():
    embeddings1, embeddings2, _ = create_embeddings([])

    faiss_db = FAISS.load_local(
        "faiss_index",
        embeddings1,
        allow_dangerous_deserialization=True
    )

    chroma_db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings2
    )

    return faiss_db, chroma_db


# 🔹 answer function
def get_answer(db, model_name, query):
    results = db.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in results])

    prompt = f"""
Answer clearly and shortly.

Context:
{context}

Question: {query}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# 🔥 MAIN
if __name__ == "__main__":
    faiss_db, chroma_db = load_dbs()

    while True:
        query = input("\nAsk: ")

        if query.lower() == "exit":
            break

        try:
            # 🔹 config 1
            ans1 = get_answer(
                faiss_db,
                "llama-3.1-8b-instant",
                query
            )

            # 🔹 config 2 (FIXED)
            ans2 = get_answer(
                chroma_db,
                "llama-3.1-8b-instant",
                query
            )

            print("\n🔹 FAISS + MiniLM + LLaMA:\n", ans1)
            print("\n🔹 Chroma + MPNet + LLaMA-70B:\n", ans2)

        except Exception as e:
            print("❌ ERROR:", e)