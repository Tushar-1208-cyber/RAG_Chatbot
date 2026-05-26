from loader import load_docs
from chunking import split_docs
from embedding import get_embeddings
from vector_db import create_faiss
from llm import get_llm

print("🚀 Program start ho gaya")

try:
    print("📄 Loading PDFs...")
    docs1 = load_docs("data/dbms.pdf")
    docs2 = load_docs("data/os.pdf")
    docs3 = load_docs("data/cn.pdf")

    docs = docs1 + docs2 + docs3

    print("✂️ Chunking...")
    chunks = split_docs(docs)

    print("🔢 Embedding...")
    embeddings = get_embeddings()

    print("🗄️ Creating DB...")
    db = create_faiss(chunks, embeddings)

    print("🔍 Retriever ready...")
    
    # 🔥 IMPROVED RETRIEVER (important)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    print("🤖 Loading LLM...")
    llm = get_llm()

    print("✅ Ready! Ask question now\n")

    while True:
        query = input("Ask question: ")

        # 🔥 SMART QUERY BOOST
        better_query = query + " definition explanation in DBMS database computer networks operating system"

        retrieved_docs = retriever.invoke(better_query)

        # 🔍 DEBUG (optional but useful)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        print("\n🔎 Retrieved Context (preview):\n", context[:300])

        # 🔥 IMPROVED PROMPT
        prompt = f"""
You are a helpful university assistant.

Answer the question using ONLY the context below.
If answer is partially available, still explain it clearly.
If not found, say "Not found in provided documents".

Context:
{context}

Question:
{query}

Answer:
"""

        response = llm.invoke(prompt)

        print("\n📌 Answer:\n", response)

except Exception as e:
    print("❌ Error aaya:", e)