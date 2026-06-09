# 🤖 RAG Document Assistant

A fully local **Retrieval-Augmented Generation (RAG)** application that lets you ask questions about any PDF document — powered by **Ollama (Llama 3.2)** and **ChromaDB**. No API keys, no cloud, completely free to run.

---

## 🎯 What It Does

Upload any PDF and ask questions about it in plain English. The system finds the most relevant sections and generates accurate, context-aware answers.

```
PDF Document
    → Text extraction (PyPDF)
    → Split into chunks (LangChain)
    → Vector embeddings (Sentence Transformers)
    → Stored in ChromaDB
    → User asks a question
    → Semantic search finds top 6 relevant chunks
    → Llama 3.2 generates an answer
    → Answer + sources returned to user
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM (local) | Ollama — Llama 3.2 |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| PDF Loading | LangChain + PyPDF |
| RAG Framework | LangChain |
| Language | Python 3.13 |

---

## 🚀 How to Run

### 1. Install Ollama and pull the model
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

### 2. Clone the repo
```bash
git clone https://github.com/Mohamad-h-c/rag-document-assistant.git
cd rag-document-assistant
```

### 3. Create virtual environment and install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install langchain langchain-community langchain-text-splitters langchain-ollama chromadb pypdf sentence-transformers
```

### 4. Add your PDF
```bash
mkdir docs
cp your-document.pdf docs/
```

### 5. Run the assistant
```bash
python3 app.py
```

### 6. Ask questions!
```
==================================================
🤖 RAG Document Assistant — Ready!
   Ask questions about your PDF.
   Type 'exit' to quit.
==================================================

Your question: What is the main research objective?

⏳ Thinking...

💬 Answer:
The main research objective is...

📚 Sources:
   → your-document.pdf, page 3
   → your-document.pdf, page 7
```

---

## ⚙️ Configuration

You can tune the RAG parameters at the top of `app.py`:

```python
CHUNK_SIZE    = 1200   # Size of each text chunk
CHUNK_OVERLAP = 200    # Overlap between chunks
TOP_K         = 6      # Number of chunks retrieved per query
MODEL_NAME    = "llama3.2"  # Ollama model to use
```

---

## 🎓 About

Built as part of my AI/ML portfolio while completing a double M.Sc. in:
- **Artificial Intelligence** — Østfold University College (HiØ), Norway
- **IT, Digitalisation & Sustainability** — Lucerne University of Applied Sciences (HSLU), Switzerland

This project demonstrates end-to-end RAG pipeline design, vector database integration, and local LLM deployment — core skills for AI engineering roles.

---

## 👤 Author

**Mohamad Hussen Chamsi**
📫 [LinkedIn](https://www.linkedin.com/in/YOUR_LINKEDIN_HERE) | ✉️ YOUR_EMAIL_HERE
