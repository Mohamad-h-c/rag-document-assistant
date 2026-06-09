import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── CONFIG ───────────────────────────────────────────────────────────────
DOCS_FOLDER  = "docs"
MODEL_NAME   = "llama3.2"
DB_FOLDER    = "chroma_db"
CHUNK_SIZE   = 1200      # was 500 — bigger = more context
CHUNK_OVERLAP = 200      # was 50  — more overlap = fewer missed answers
TOP_K        = 6         # was 3   — fetch more chunks per query

# ── STEP 1: Load PDFs ────────────────────────────────────────────────────
def load_documents(folder):
    documents = []
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    if not pdf_files:
        print("❌ No PDF files found in /docs folder.")
        exit()
    print(f"\n📄 Found {len(pdf_files)} PDF file(s): {pdf_files}")
    for pdf_file in pdf_files:
        path = os.path.join(folder, pdf_file)
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
        print(f"   ✅ Loaded: {pdf_file}")
    return documents

# ── STEP 2: Split into chunks ─────────────────────────────────────────────
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"\n✂️  Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks

# ── STEP 3: Store in ChromaDB ─────────────────────────────────────────────
def build_vectorstore(chunks):
    # Remove old DB so we rebuild with new chunk settings
    import shutil
    if os.path.exists(DB_FOLDER):
        shutil.rmtree(DB_FOLDER)
        print("\n🗑️  Removed old vector database (rebuilding with new settings)")

    print("🔍 Building vector database...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_FOLDER
    )
    print("   ✅ Vector database ready!")
    return vectorstore

# ── STEP 4: Build RAG chain ───────────────────────────────────────────────
def build_rag_chain(vectorstore):
    llm = OllamaLLM(model=MODEL_NAME)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    prompt = ChatPromptTemplate.from_template("""
You are a precise research assistant. Your job is to answer questions 
based strictly on the provided context from a research document.

Rules:
- Answer directly and specifically
- If the answer is in the context, provide it fully
- If the context contains partial information, share what you find
- Only say "I don't have enough information" if the topic is truly absent
- Quote relevant parts when helpful

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(
            f"[Page {doc.metadata.get('page', '?') + 1}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ── STEP 5: Interactive Q&A loop ──────────────────────────────────────────
def run_qa_loop(chain, retriever):
    print("\n" + "="*50)
    print("🤖 RAG Document Assistant — Ready!")
    print(f"   Settings: chunk_size={CHUNK_SIZE}, top_k={TOP_K}")
    print("   Type 'exit' to quit.")
    print("="*50 + "\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("\n👋 Goodbye!")
            break
        if not question:
            continue

        print("\n⏳ Thinking...\n")
        answer = chain.invoke(question)
        print(f"💬 Answer:\n{answer}")

        docs = retriever.invoke(question)
        print("\n📚 Sources:")
        for doc in docs:
            page = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "unknown")
            print(f"   → {os.path.basename(source)}, page {page + 1}")
        print("\n" + "-"*50 + "\n")

# ── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Starting RAG Document Assistant (improved settings)...")
    documents        = load_documents(DOCS_FOLDER)
    chunks           = split_documents(documents)
    vectorstore      = build_vectorstore(chunks)
    chain, retriever = build_rag_chain(vectorstore)
    run_qa_loop(chain, retriever)
