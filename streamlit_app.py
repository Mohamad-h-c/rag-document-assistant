import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── CONFIG ───────────────────────────────────────────────────────────────
MODEL_NAME    = "llama3.2"
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200
TOP_K         = 6

# ── PAGE SETUP ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Document Assistant")
st.caption("Upload a PDF and ask questions about it — powered by Llama 3.2 (local, free)")

# ── SESSION STATE ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ── SIDEBAR: PDF UPLOAD ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        if st.button("🚀 Process PDF", use_container_width=True):
            with st.spinner("Reading and indexing PDF..."):

                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Load
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()

                # Split
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                chunks = splitter.split_documents(documents)

                # Embed + store
                embeddings = SentenceTransformerEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": TOP_K}
                )

                # Build chain
                llm = OllamaLLM(model=MODEL_NAME)
                prompt = ChatPromptTemplate.from_template("""
You are a precise research assistant. Answer the question based strictly on the context below.
If the answer is in the context, provide it fully.
Only say "I don't have enough information" if the topic is truly absent.

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

                st.session_state.chain = chain
                st.session_state.retriever = retriever
                st.session_state.messages = []
                os.unlink(tmp_path)

            st.success(f"✅ Ready! {len(chunks)} chunks indexed.")
            st.info(f"📄 {uploaded_file.name}")

    st.divider()
    st.markdown("**Settings**")
    st.caption(f"Model: `{MODEL_NAME}`")
    st.caption(f"Chunk size: `{CHUNK_SIZE}`")
    st.caption(f"Top K: `{TOP_K}`")

    if st.session_state.chain:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ── MAIN: CHAT INTERFACE ─────────────────────────────────────────────────
if not st.session_state.chain:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.markdown("""
    **How it works:**
    1. Upload any PDF document
    2. Click **Process PDF**
    3. Ask questions in the chat
    4. Get answers with page references
    """)
else:
    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("Ask a question about your document..."):

        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(question)
                docs   = st.session_state.retriever.invoke(question)

                # Sources
                sources = []
                for doc in docs:
                    page   = doc.metadata.get("page", "?")
                    source = doc.metadata.get("source", "document")
                    sources.append(f"Page {int(page) + 1}")

                unique_sources = list(dict.fromkeys(sources))
                source_text = " · ".join(unique_sources)

                full_response = f"{answer}\n\n📚 *Sources: {source_text}*"
                st.markdown(full_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
