import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama

load_dotenv()

DATA_DIR = Path("data")
VSTORE_DIR = DATA_DIR / "vectorstore"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def load_vectorstore():
    """
    Load the FAISS vector store that we built from Blackboard assignments.
    """
    if not VSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {VSTORE_DIR}. Run build_index.py first."
        )

    embedding = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vstore = FAISS.load_local(
        str(VSTORE_DIR),
        embeddings=embedding,
        allow_dangerous_deserialization=True,
    )
    return vstore


def retrieve_assignments(query: str, k: int = 10):
    """
    Retrieve the top-k most relevant assignments from the vector store.
    Compatible with newer LangChain retriever API.
    """
    vstore = load_vectorstore()
    retriever = vstore.as_retriever(search_kwargs={"k": k})

    # Newer retrievers use .invoke() instead of get_relevant_documents()
    docs = retriever.invoke(query)
    return docs


def build_prompt(query: str, docs):
    """
    Build a prompt that includes the retrieved assignments as context
    and asks the LLM to create a study plan.
    """
    today = datetime.now().strftime("%B %d, %Y")

    if not docs:
        context = "No assignments were found in the system."
    else:
        lines = []
        for d in docs:
            meta = d.metadata or {}
            course = meta.get("course", "Unknown Course")
            title = meta.get("title", "")
            due = meta.get("due", "Unknown due date")
            # If metadata isn't populated, fall back to page_content
            if not title and d.page_content:
                title = d.page_content
            lines.append(f"- {course}: {title} (Due: {due})")

        context = "\n".join(lines)

    system_prompt = (
        "You are an AI study planner for a college student taking multiple courses.\n"
        "You receive context about assignments (course names, titles, and due info).\n"
        "Analyze the tasks, prioritize them by urgency and importance,\n"
        "and produce an optimized daily schedule for the next 7 days.\n"
        "Mention which tasks are high priority and explain briefly why.\n"
        "Assume the student has 2–4 hours per day.\n"
        "Be specific but concise, using bullet points grouped by day.\n"
    )

    full_prompt = f"""
{system_prompt}

Today's date is {today}.

Here are the upcoming or relevant assignments:

{context}

User question:
\"\"\"{query}\"\"\"

Now, based on the above context and the user's question, create a clear, realistic 7-day study plan.
"""

    return full_prompt


def generate_study_plan(query: str) -> str:
    """
    Run the full RAG-style flow:
      - retrieve relevant assignments
      - build a prompt with context
      - call the local LLM via Ollama
    """
    docs = retrieve_assignments(query, k=10)
    prompt = build_prompt(query, docs)

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
    )

    response = llm.invoke(prompt)

    # ChatOllama may return an AIMessage object or a raw string
    if hasattr(response, "content"):
        return response.content
    return str(response)


def main():
    print("🧠 AI-Powered Study Planner")
    print("---------------------------")
    print("Examples:")
    print("- What should I work on tonight?")
    print("- How should I plan my week?")
    print("- Which assignments are most urgent?")
    print()

    user_q = input("Enter your question about your assignments: ").strip()
    if not user_q:
        user_q = "Create an optimized 7-day study schedule based on my upcoming assignments."

    print("\n📚 Generating study plan with FAISS + Ollama...\n")
    try:
        plan = generate_study_plan(user_q)
        print(plan)
    except Exception as e:
        print(f"❌ Error while generating plan: {e}")


if __name__ == "__main__":
    main()
