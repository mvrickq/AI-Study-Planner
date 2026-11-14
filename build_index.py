import json
from pathlib import Path

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = Path("data")
ASSIGNMENTS_JSON = DATA_DIR / "assignments.json"
VSTORE_DIR = DATA_DIR / "vectorstore"


def main():
    if not ASSIGNMENTS_JSON.exists():
        raise FileNotFoundError(
            f"{ASSIGNMENTS_JSON} not found. Run fetch_assignments.py first."
        )

    with ASSIGNMENTS_JSON.open("r", encoding="utf-8") as f:
        assignments = json.load(f)

    if not assignments:
        print("No assignments found to index.")
        return

    texts = []
    metas = []

    for a in assignments:
        text = f"{a['course']}: {a['title']} (Due: {a['due']})"
        texts.append(text)
        metas.append(a)

    print(f"Embedding {len(texts)} assignments with Sentence-Transformers...")

    embedding = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build FAISS vector store via LangChain
    vstore = FAISS.from_texts(texts=texts, embedding=embedding, metadatas=metas)

    VSTORE_DIR.mkdir(exist_ok=True)
    vstore.save_local(str(VSTORE_DIR))
    print(f"✅ Saved FAISS vector store to {VSTORE_DIR}")


if __name__ == "__main__":
    main()
