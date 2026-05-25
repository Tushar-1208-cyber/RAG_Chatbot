from langchain_community.embeddings import HuggingFaceEmbeddings
from loader import load_docs
from chunking import split_docs

def create_embeddings(chunks):
    # 🔹 Embedding 1 (MiniLM - fast)
    embeddings1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 🔹 Embedding 2 (MPNet - more accurate)
    embeddings2 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    return embeddings1, embeddings2, chunks


# testing
if __name__ == "__main__":
    docs = load_docs("data/os.pdf")
    chunks = split_docs(docs)

    embeddings1, embeddings2, chunks = create_embeddings(chunks)

    print("✅ Embedding 1 (MiniLM) ready")
    print("✅ Embedding 2 (MPNet) ready")
    print("Total chunks:", len(chunks))