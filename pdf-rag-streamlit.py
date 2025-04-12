import streamlit as st
import os
import logging
import time
import fitz  # PyMuPDF
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# --- Logging ---
logging.basicConfig(level=logging.INFO)

# --- Constants ---
DOC_PATH = ".\data\ipc_law.pdf"
MODEL_NAME = "ipc"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"

# --- Vector DB Setup (Optional) ---
def ingest_pdf(doc_path):
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        return loader.load()
    return None

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    return splitter.split_documents(documents)

@st.cache_resource
def load_vector_db():
    ollama.pull(EMBEDDING_MODEL)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    if os.path.exists(PERSIST_DIRECTORY):
        return Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
    else:
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None
        chunks = split_documents(data)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        return vector_db

# --- UI Setup ---
st.set_page_config(page_title="The Virtual Suits", page_icon="images/favicon.png", layout="centered")

st.markdown("""
    <style>
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .chat-row {
        display: flex;
        align-items: flex-start;
        margin: 12px 0;
    }
    .chat-message {
        padding: 12px 16px;
        border-radius: 16px;
        font-size: 16px;
        line-height: 1.5;
        max-width: 85%;
        word-wrap: break-word;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    .user-message {
        background-color: #d1f5d3;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .assistant-message {
        background-color: #f0f0f5;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    .icon {
        width: 36px;
        height: 36px;
        margin: 0 8px;
        border-radius: 50%;
        background-color: #ddd;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 18px;
    }
    .user-icon {
        background-color: #84e1a7;
        color: white;
    }
    .bot-icon {
        background-color: #c4c4f7;
        color: white;
    }
    .centered-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: black;
    }
    </style>
    <div class="centered-title">⚖️ The Virtual Suits ⚖️</div>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# --- Sidebar File Upload ---
st.sidebar.header("Upload Legal Documents Here")
uploaded_file = st.sidebar.file_uploader("Upload PDF document", type=["pdf"])

# --- PDF Handling ---
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def summarize_document(text):
    prompt = f"""
    Analyze this legal document and provide a concise summary with key legal points in bullet form:

    {text}

    Format:
    - 📌 Description
    - 📌 Key points about the legal Document
    - 📌 Additional Information
    ...
    """
    return llm.invoke(prompt).content

# --- LLM Init ---
llm = ChatOllama(model=MODEL_NAME)

# --- If File Uploaded, Process It Once ---
if uploaded_file and not st.session_state.file_uploaded:
    extracted_text = extract_text_from_pdf(uploaded_file)

    with st.sidebar:
        analyzing_status = st.empty()
        analyzing_status.info("🔍 Analyzing the document...")

        summary = summarize_document(extracted_text)

        analyzing_status.empty()  # Remove message after done

    summary_msg = "**📄 Summary of Uploaded Document:**\n\n" + summary
    st.session_state.chat_history.append({"role": "assistant", "content": summary_msg})
    st.session_state.file_uploaded = True

# --- Input Box ---
user_query = st.chat_input("Type your message...")

# --- Handle Input First ---
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    try:
        full_response = llm.invoke(user_query).content
    except Exception as e:
        full_response = f"⚠️ Error: {str(e)}"
    st.session_state.chat_history.append({"role": "assistant", "content": ""})

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history[:-1]:
        role, content = msg["role"], msg["content"]
        if role == "user":
            st.markdown(f"""
                <div class="chat-row" style="justify-content: flex-end;">
                    <div class="chat-message user-message">{content}</div>
                    <div class="icon user-icon">🧑</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-row" style="justify-content: flex-start;">
                    <div class="icon bot-icon">🤖</div>
                    <div class="chat-message assistant-message">{content}</div>
                </div>
            """, unsafe_allow_html=True)

    # Stream effect
    bot_container = st.empty()
    typed = ""
    for char in full_response:
        typed += char
        bot_container.markdown(f"""
            <div class="chat-row" style="justify-content: flex-start;">
                <div class="icon bot-icon">🤖</div>
                <div class="chat-message assistant-message">{typed}</div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(0.015)
    st.session_state.chat_history[-1]["content"] = full_response
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        role, content = msg["role"], msg["content"]
        if role == "user":
            st.markdown(f"""
                <div class="chat-row" style="justify-content: flex-end;">
                    <div class="chat-message user-message">{content}</div>
                    <div class="icon user-icon">🧑</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-row" style="justify-content: flex-start;">
                    <div class="icon bot-icon">🤖</div>
                    <div class="chat-message assistant-message">{content}</div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
