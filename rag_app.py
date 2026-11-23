"""
Multi-Document RAG Application with Conversational Memory & Source Citations
Built with Streamlit, LangChain, LangGraph, and Google/Groq APIs.
"""

import os
import io
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv, find_dotenv

import streamlit as st
from pypdf import PdfReader
import docx

# LangChain & LangGraph imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(find_dotenv(usecwd=True))

# ==============================================================================
# 1. STREAMLIT CONFIGURATION & CUSTOM STYLES
# ==============================================================================
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Main container styling */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }
    /* Header badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.4rem;
    }
    .badge-pdf { background-color: #fee2e2; color: #991b1b; }
    .badge-docx { background-color: #dbeafe; color: #1e40af; }
    .badge-txt { background-color: #fef3c7; color: #92400e; }
    .badge-chunk { background-color: #e0e7ff; color: #3730a3; }
    
    /* Citation box */
    .citation-card {
        border-left: 3px solid #3b82f6;
        background-color: #f8fafc;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.375rem 0.375rem 0;
        font-size: 0.88rem;
    }
    /* Dark mode support for citations */
    @media (prefers-color-scheme: dark) {
        .citation-card {
            background-color: #1e293b;
            border-left-color: #60a5fa;
            color: #e2e8f0;
        }
        .badge-pdf { background-color: #450a0a; color: #fca5a5; }
        .badge-docx { background-color: #172554; color: #93c5fd; }
        .badge-txt { background-color: #451a03; color: #fcd34d; }
    }
    /* Metric pill */
    .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.75rem;
        background: #f1f5f9;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. DOCUMENT LOADERS & PARSERS
# ==============================================================================
def load_single_file(file_name: str, file_bytes: bytes) -> List[Document]:
    """Extracts text and page metadata from PDF, DOCX, or TXT file bytes."""
    ext = file_name.split(".")[-1].lower()
    documents = []

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_name,
                            "page": page_idx + 1,
                            "file_type": "PDF",
                            "total_pages": len(reader.pages),
                        },
                    )
                )

    elif ext == "docx":
        doc = docx.Document(io.BytesIO(file_bytes))
        # Group paragraphs into meaningful sections
        current_section = []
        section_idx = 1
        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            current_section.append(text)
            if len("\n".join(current_section)) > 1000:
                documents.append(
                    Document(
                        page_content="\n\n".join(current_section),
                        metadata={
                            "source": file_name,
                            "page": section_idx,
                            "file_type": "DOCX",
                        },
                    )
                )
                current_section = []
                section_idx += 1
        if current_section:
            documents.append(
                Document(
                    page_content="\n\n".join(current_section),
                    metadata={
                        "source": file_name,
                        "page": section_idx,
                        "file_type": "DOCX",
                    },
                )
            )

    elif ext == "txt":
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = file_bytes.decode("latin-1", errors="ignore")

        content = content.strip()
        if content:
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file_name,
                        "page": 1,
                        "file_type": "TXT",
                    },
                )
            )

    return documents


def process_uploaded_files(
    uploaded_files, chunk_size: int, chunk_overlap: int, google_api_key: str
) -> Optional[InMemoryVectorStore]:
    """Parses files, splits into chunks, embeds with Gemini Embeddings, and creates an InMemoryVectorStore."""
    all_raw_docs = []
    file_summaries = []

    for file in uploaded_files:
        if hasattr(file, "read"):
            content = file.read()
            name = file.name
            size = len(content)
        else:
            # Handle local file path if passing path
            with open(file, "rb") as f:
                content = f.read()
            name = os.path.basename(file)
            size = len(content)

        docs = load_single_file(name, content)
        if docs:
            all_raw_docs.extend(docs)
            file_summaries.append({
                "name": name,
                "type": name.split(".")[-1].upper(),
                "size_kb": round(size / 1024, 1),
                "sections": len(docs),
            })

    if not all_raw_docs:
        return None

    # Chunking using RecursiveCharacterTextSplitter (matching notebooks)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(all_raw_docs)

    # Attach unique chunk ids
    for idx, c in enumerate(chunks):
        c.metadata["chunk_id"] = idx + 1

    # Embeddings using gemini-embedding-001 (verified working)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key,
    )
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)

    # Update session state summaries
    st.session_state["file_summaries"] = file_summaries
    st.session_state["total_chunks"] = len(chunks)
    st.session_state["raw_docs_count"] = len(all_raw_docs)

    return vector_store


# ==============================================================================
# 3. RAG GRAPH IMPLEMENTATION (LANGGRAPH)
# ==============================================================================
class RagState(BaseModel):
    query: str
    standalone_query: str = ""
    history: List[Dict[str, str]] = Field(default_factory=list)
    documents: List[Any] = Field(default_factory=list)
    context: str = ""
    ans: str = ""
    citations: List[Dict[str, Any]] = Field(default_factory=list)


def create_rag_graph(vector_store: InMemoryVectorStore, llm: Any, top_k: int = 4):
    """Compiles a LangGraph RAG pipeline with query contextualization, retrieval, and source citation."""

    def contextualize_node(state: RagState) -> Dict[str, Any]:
        """Rewrites follow-up questions into standalone queries based on conversational memory."""
        if not state.history:
            return {"standalone_query": state.query}

        # Keep recent 4 turns to avoid context overflow
        formatted_history = "\n".join(
            [f"{m['role'].capitalize()}: {m['content']}" for m in state.history[-4:]]
        )
        prompt = f"""You are an AI assistant. Given the conversation history below and a follow-up question, rephrase the follow-up question into a standalone search query that can be understood without the chat history.
Do NOT answer the question. Only output the rephrased standalone query. If the question is already standalone, return it unchanged.

Conversation History:
{formatted_history}

Follow-up Question: {state.query}
Standalone Query:"""
        try:
            res = llm.invoke(prompt)
            standalone = res.content.strip()
            return {"standalone_query": standalone}
        except Exception:
            return {"standalone_query": state.query}

    def retrieve_node(state: RagState) -> Dict[str, Any]:
        """Retrieves top_k chunks from the vector store."""
        search_query = state.standalone_query if state.standalone_query else state.query
        docs = vector_store.similarity_search(query=search_query, k=top_k)
        return {"documents": docs}

    def create_context_node(state: RagState) -> Dict[str, Any]:
        """Formats context for the LLM and tracks citations."""
        context_parts = []
        citations = []

        for idx, doc in enumerate(state.documents):
            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", 1)
            ftype = doc.metadata.get("file_type", "DOC")
            chunk_id = doc.metadata.get("chunk_id", idx + 1)

            header = f"[Source #{idx+1}: {source} (Page/Part {page})]"
            context_parts.append(f"{header}\n{doc.page_content}")

            citations.append({
                "ref_id": idx + 1,
                "source": source,
                "page": page,
                "file_type": ftype,
                "chunk_id": chunk_id,
                "snippet": doc.page_content,
            })

        full_context = "\n\n".join(context_parts)
        return {"context": full_context, "citations": citations}

    def generate_node(state: RagState) -> Dict[str, Any]:
        """Generates answer strictly grounded in the context with explicit citations."""
        history_context = ""
        if state.history:
            history_context = "\nRecent Conversation:\n" + "\n".join(
                [f"{m['role'].capitalize()}: {m['content']}" for m in state.history[-2:]]
            ) + "\n"

        system_prompt = f"""You are an accurate, helpful AI document question-answering assistant.
Your task is to answer the user's question using ONLY the retrieved document context provided below.

Strict Citation Rules:
1. Every major fact, statement, or data point MUST be cited using inline brackets: [Source: <filename>, Page: <page>] (e.g. [Source: report.pdf, Page: 3]).
2. If the answer cannot be found or deduced from the provided context, state clearly: "I cannot find the answer to this question in the uploaded documents."
3. Do not make up information or cite documents that are not in the context.
4. Keep the answer structured, well-formatted, and concise.

Document Context:
{state.context}
{history_context}
User Question: {state.query}
Answer:"""

        res = llm.invoke(system_prompt)
        return {"ans": res.content}

    # Graph compilation
    builder = StateGraph(RagState)
    builder.add_node("contextualize", contextualize_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("create_context", create_context_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "retrieve")
    builder.add_edge("retrieve", "create_context")
    builder.add_edge("create_context", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


# ==============================================================================
# 4. INITIALIZE SESSION STATE
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "file_summaries" not in st.session_state:
    st.session_state.file_summaries = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "raw_docs_count" not in st.session_state:
    st.session_state.raw_docs_count = 0


# ==============================================================================
# 5. SIDEBAR: DOCUMENT UPLOAD & SETTINGS
# ==============================================================================
with st.sidebar:
    st.title("📚 Document Hub")
    st.caption("Upload & query PDF, DOCX, and TXT files with citations.")

    # --- Section A: File Upload ---
    st.subheader("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, DOCX, TXT):",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload one or multiple documents to query simultaneously.",
    )

    # Sample data quick-load button
    sample_files_available = [
        "data/Record.pdf",
        "data/speech.txt",
        "data/sample_test.docx",
    ]
    existing_samples = [f for f in sample_files_available if os.path.exists(f)]

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        process_btn = st.button("⚡ Index Uploaded", type="primary", use_container_width=True)
    with col_btn2:
        load_samples_btn = st.button("📁 Load Samples", use_container_width=True)

    # --- Section B: Configuration & Keys ---
    with st.expander("⚙️ Model & Retrieval Settings", expanded=False):
        provider = st.selectbox(
            "LLM Provider:",
            options=["Groq (Ultra-Fast)", "Google Gemini"],
            index=0,
        )

        env_groq_key = os.getenv("GROQ_API_KEY", "")
        env_google_key = os.getenv("GOOGLE_API_KEY", "")

        groq_api_key = st.text_input(
            "Groq API Key:",
            value=env_groq_key,
            type="password",
            help="Required for Groq LLM inference.",
        )

        google_api_key = st.text_input(
            "Google API Key:",
            value=env_google_key,
            type="password",
            help="Required for Gemini Embeddings.",
        )

        if provider == "Groq (Ultra-Fast)":
            model_name = st.selectbox(
                "Model:",
                options=["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
                index=0,
            )
        else:
            model_name = st.selectbox(
                "Model:",
                options=["gemini-2.5-flash", "gemini-3.8-flash"],
                index=0,
            )

        top_k = st.slider("Top Chunks (k):", min_value=1, max_value=8, value=4, step=1)
        chunk_size = st.slider("Chunk Size:", min_value=200, max_value=1500, value=600, step=100)
        chunk_overlap = st.slider("Chunk Overlap:", min_value=0, max_value=300, value=100, step=20)
        temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    # --- Section C: Indexing Logic ---
    files_to_process = None
    if process_btn:
        if not uploaded_files:
            st.warning("Please select at least one file to upload first.")
        else:
            files_to_process = uploaded_files

    if load_samples_btn:
        if existing_samples:
            files_to_process = existing_samples
        else:
            st.warning("No sample files found in the data/ directory.")

    if files_to_process:
        if not google_api_key:
            st.error("Google API Key is required for Gemini vector embeddings. Please provide it in settings or .env.")
        else:
            with st.spinner("Processing & indexing documents..."):
                try:
                    v_store = process_uploaded_files(
                        files_to_process,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        google_api_key=google_api_key,
                    )
                    if v_store:
                        st.session_state.vector_store = v_store
                        st.success(f"Successfully indexed {len(st.session_state.file_summaries)} file(s) into {st.session_state.total_chunks} chunks!")
                    else:
                        st.error("No valid text could be extracted from the uploaded files.")
                except Exception as e:
                    st.error(f"Error indexing files: {e}")

    # --- Section D: Active Knowledge Base Stats ---
    st.divider()
    st.subheader("📊 Knowledge Base")
    if st.session_state.vector_store and st.session_state.file_summaries:
        st.markdown(
            f"""
            <div style="margin-bottom: 0.5rem;">
                <span class="metric-pill">📄 <b>{len(st.session_state.file_summaries)}</b> Files</span>
                <span class="metric-pill">🧩 <b>{st.session_state.total_chunks}</b> Chunks</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for f in st.session_state.file_summaries:
            badge_class = f"badge-{f['type'].lower()}"
            st.markdown(
                f"""
                <div style="font-size: 0.86rem; padding: 0.3rem 0; border-bottom: 1px solid #f1f5f9;">
                    <span class="badge {badge_class}">{f['type']}</span>
                    <b>{f['name']}</b> ({f['size_kb']} KB)
                </div>
                """,
                unsafe_allow_html=True,
            )

        col_clear1, col_clear2 = st.columns(2)
        with col_clear1:
            if st.button("🗑️ Reset Docs", use_container_width=True):
                st.session_state.vector_store = None
                st.session_state.file_summaries = []
                st.session_state.total_chunks = 0
                st.session_state.messages = []
                st.rerun()
        with col_clear2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    else:
        st.info("No documents indexed yet. Upload documents above or click 'Load Samples'.")


# ==============================================================================
# 6. MAIN CHAT INTERFACE
# ==============================================================================
st.title("📚 Multi-Document RAG Assistant")
st.markdown("Ask questions across multiple PDF, DOCX, and TXT files with **conversational memory** and **source citations**.")

# Check LLM Initialization
llm = None
if provider == "Groq (Ultra-Fast)":
    if groq_api_key:
        llm = ChatGroq(model=model_name, groq_api_key=groq_api_key, temperature=temperature)
    else:
        st.warning("⚠️ Groq API key is missing. Please add it to the sidebar settings or .env file.")
else:
    if google_api_key:
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=google_api_key, temperature=temperature)
    else:
        st.warning("⚠️ Google API key is missing. Please add it to the sidebar settings or .env file.")

# Welcome banner when no documents are loaded
if not st.session_state.vector_store:
    st.markdown(
        """
        ---
        ### 🚀 Getting Started
        1. **Upload Documents** using the sidebar on the left (PDF, DOCX, TXT supported).
        2. Click **⚡ Index Uploaded** (or click **📁 Load Samples** to test immediately with the included sample files).
        3. Type your questions below. The assistant will retrieve relevant excerpts, remember conversational context, and cite exact sources!
        """
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📄 **Multi-Format Support**\nSeamlessly extracts text from PDFs with pages, DOCX with sections, and plain text files.")
    with col2:
        st.info("🧠 **Conversational Memory**\nMaintains chat history and contextually resolves follow-up queries.")
    with col3:
        st.info("🏷️ **Source Citations**\nInspect exact passages, filenames, and page numbers backing every generated answer.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If there are citation sources attached to this message, display them in an expander
        citations = msg.get("citations", [])
        if citations:
            with st.expander(f"📖 Sources & Citations ({len(citations)} references)", expanded=False):
                for c in citations:
                    badge_class = f"badge-{c['file_type'].lower()}"
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <span class="badge {badge_class}">{c['file_type']}</span>
                            <b>{c['source']}</b> — Page/Part: <code>{c['page']}</code> (Chunk #{c['chunk_id']})
                            <div style="margin-top: 0.4rem; font-style: italic;">"{c['snippet']}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Chat Input & Response Generation
user_query = st.chat_input("Ask a question about your uploaded documents...")

if user_query:
    if not st.session_state.vector_store:
        st.error("Please upload and index documents first using the sidebar.")
    elif not llm:
        st.error("Please provide a valid API key in the sidebar settings to generate responses.")
    else:
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Build and invoke RAG Graph
        with st.chat_message("assistant"):
            with st.spinner("Searching documents & generating cited answer..."):
                try:
                    rag_graph = create_rag_graph(
                        vector_store=st.session_state.vector_store,
                        llm=llm,
                        top_k=top_k,
                    )

                    # Prepare conversation history for memory node
                    history_for_state = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]

                    # Execute graph
                    output = rag_graph.invoke({
                        "query": user_query,
                        "history": history_for_state,
                    })

                    ans_text = output.get("ans", "No response generated.")
                    citations = output.get("citations", [])
                    standalone_q = output.get("standalone_query", "")

                    # Render answer
                    st.markdown(ans_text)

                    # Display citation expander
                    if citations:
                        with st.expander(f"📖 Sources & Citations ({len(citations)} references)", expanded=True):
                            if standalone_q and standalone_q != user_query:
                                st.caption(f"🔍 *Contextualized query used for retrieval:* **{standalone_q}**")
                            for c in citations:
                                badge_class = f"badge-{c['file_type'].lower()}"
                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        <span class="badge {badge_class}">{c['file_type']}</span>
                                        <b>{c['source']}</b> — Page/Part: <code>{c['page']}</code> (Chunk #{c['chunk_id']})
                                        <div style="margin-top: 0.4rem; font-style: italic;">"{c['snippet']}"</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Save to session history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ans_text,
                        "citations": citations,
                    })

                except Exception as e:
                    st.error(f"Error during RAG execution: {str(e)}")
