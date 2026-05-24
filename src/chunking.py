from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_docs

def split_docs(docs):
    # 🔹 Chunking Strategy 1 (small chunks)
    splitter1 = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    # 🔹 Chunking Strategy 2 (large chunks)
    splitter2 = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks1 = splitter1.split_documents(docs)
    chunks2 = splitter2.split_documents(docs)

    return chunks1, chunks2


# testing
if __name__ == "__main__":
    docs = load_docs("data/os.pdf")
    chunks1, chunks2 = split_docs(docs)

    print("✅ Small chunks:", len(chunks1))
    print("✅ Large chunks:", len(chunks2))

    print("\nSample Small Chunk:\n", chunks1[0].page_content[:200])
    print("\nSample Large Chunk:\n", chunks2[0].page_content[:200])