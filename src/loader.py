import os
from langchain_community.document_loaders import PyPDFLoader

def load_docs(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    return documents


# testing
if __name__ == "__main__":
    docs = load_docs("data")   # 👈 folder path (multiple PDFs)

    print("Total pages loaded:", len(docs))
    print("\nSample content:\n", docs[0].page_content[:200])