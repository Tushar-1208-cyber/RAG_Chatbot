from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db

if __name__ == "__main__":
    db = load_db()

    while True:
        query = input("\nAsk: ")

        if query.lower() == "exit":
            break

        results = db.similarity_search(query, k=3)

        answer = ""

        # 🔥 check all 3 results
        for doc in results:
            lines = doc.page_content.split("\n")

            for line in lines:
                if "process is" in line.lower() or "process is a" in line.lower():
                    answer = line
                    break

            if answer:
                break

        # fallback
        if not answer:
            answer = results[0].page_content.split("\n")[0]

        print("\nAnswer:", answer.strip())