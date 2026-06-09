# 🤖 RAG Document Assistant

A fully local **Retrieval-Augmented Generation (RAG)** application that lets you ask questions about any PDF document — powered by **Ollama (Llama 3.2)** and **ChromaDB**. No API keys, no cloud, completely free to run.

Available in two modes:
- 🖥️ **Terminal app** — lightweight, runs in the command line
- 🌐 **Web app** — upload PDF and chat in the browser (Streamlit)

---

## 🌐 Web App Preview

Upload a PDF → Ask questions → Get answers with page references

```
┌─────────────────────────────────────────────────┐
│  📄 Upload PDF          │  💬 Chat               │
│  ─────────────────      │  ──────────────────    │
│  [ Choose file ]        │  You: What is the      │
│                         │  main research goal?   │
│  [ 🚀 Process PDF ]     │                        │
│                         │  🤖 The main goal is.. │
│  ─────────────────      │  📚 Sources: Page 3·7  │
│  Settings               │                        │
│  Model: llama3.2        │  [ Ask a question... ] │
│  Chunk size: 1200       │                        │
└─────────────────────────────────────────────────┘
```

---

## 🎯 How It Works

```
PDF Document
    → Text extraction (PyPDF)
    → Split into chunks (LangChain)
    → Vector embeddings (Sentence Transformers)
    → Stored in ChromaDB
    → User asks a question
    → Semantic search finds top 6 relevant chunks
    → Llama 3.2 generates an answer
    → Answer + page sources returned to user
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
| Web Interface | Streamlit |
| Language | Python 3.13 |

---

## 🚀 Getting Started

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
pip install langchain langchain-community langchain-text-splitters langchain-ollama chromadb pypdf sentence-transformers streamlit
```

---

## 🖥️ Run Terminal App

```bash
mkdir docs
cp your-document.pdf docs/
python3 app.py
```

**Example session:**
```
🚀 Starting RAG Document Assistant...

📄 Found 1 PDF file(s): ['your-document.pdf']
   ✅ Loaded: your-document.pdf
✂️  Split into 84 chunks
🔍 Building vector database...
   ✅ Vector database ready!

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

## 🌐 Run Web App

```bash
streamlit run streamlit_app.py
```

Opens automatically at `http://localhost:8501`

1. Upload any PDF in the sidebar
2. Click **Process PDF**
3. Ask questions in the chat
4. Get answers with page references

> **Note:** Make sure Ollama is running in the background:
> ```bash
> ollama serve
> ```

---

## ⚙️ Configuration

Tune RAG parameters at the top of `app.py` or `streamlit_app.py`:

```python
CHUNK_SIZE    = 1200   # Size of each text chunk (larger = more context)
CHUNK_OVERLAP = 200    # Overlap between chunks (reduces missed answers)
TOP_K         = 6      # Number of chunks retrieved per query
MODEL_NAME    = "llama3.2"  # Ollama model to use
```

---

## 🎓 About

Built as part of my AI/ML engineering portfolio while completing a double M.Sc. in:
- **Artificial Intelligence** — Østfold University College (HiØ), Norway
- **IT, Digitalisation & Sustainability** — Lucerne University of Applied Sciences (HSLU), Switzerland

This project demonstrates:
- ✅ End-to-end RAG pipeline design
- ✅ Vector database integration (ChromaDB)
- ✅ Local LLM deployment (Ollama)
- ✅ Web application development (Streamlit)
- ✅ Python software engineering best practices

---

## 👤 Author

**Mohamad Chamsi**
📫 [LinkedIn](https://www.linkedin.com/in/mohamad-chamsi-5878772b9) | ✉️ mohamad.chamsi@outlook.com
