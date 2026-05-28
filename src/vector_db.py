from langchain_community.vectorstores import FAISS, Chroma
from embedding import create_embeddings
from loader import load_docs
from chunking import split_docs


def create_vector_db():
    # load docs (MULTI PDF)
    docs = load_docs("data")

    # chunking (2 types)
    chunks1, chunks2 = split_docs(docs)

    # embeddings
    embeddings1, embeddings2, _ = create_embeddings(chunks1)

    # 🔹 FAISS (fast)
    faiss_db = FAISS.from_documents(chunks1, embeddings1)
    faiss_db.save_local("faiss_index")

    # 🔹 CHROMA (accurate)
    chroma_db = Chroma.from_documents(
        chunks2,
        embeddings2,
        persist_directory="chroma_db"
    )
    chroma_db.persist()

    print("✅ FAISS + CHROMA created successfully")


# ✅ LOAD FUNCTIONS (FIXED)
def load_faiss():
    from embedding import create_embeddings
    embeddings1, _, _ = create_embeddings([])

    return FAISS.load_local(
        "faiss_index",
        embeddings1,
        allow_dangerous_deserialization=True
    )


def load_chroma():
    from embedding import create_embeddings
    _, embeddings2, _ = create_embeddings([])

    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings2
    )


# testing
if __name__ == "__main__":
    create_vector_db()