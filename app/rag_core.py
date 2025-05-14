import streamlit as st
import os
# Updated import for HuggingFaceEmbeddings:
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "documents")
# Ensure this matches the PDF you are testing with
SOURCE_DOCUMENT_FILENAME = "Company-10k-18pages.pdf" 
SOURCE_DOCUMENT_PATH = os.path.join(DOCUMENTS_DIR, SOURCE_DOCUMENT_FILENAME)
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_store_data")
VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Embedding Model ---
@st.cache_resource
def get_embeddings_model():
    """Loads and returns the HuggingFace embeddings model."""
    try:
        print("DEBUG: Attempting to load embedding model...")
        model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print(f"DEBUG: Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        print(f"ERROR DEBUG: Error loading embedding model: {e}")
        return None

# --- Vector Store ---
def _load_and_split_pdfs(pdf_paths):
    """Loads and splits multiple PDF documents into chunks."""
    all_document_chunks = []
    print(f"DEBUG: _load_and_split_pdfs called with paths: {pdf_paths}")
    for doc_path in pdf_paths:
        if os.path.exists(doc_path):
            try:
                print(f"DEBUG: Loading PDF: {doc_path}")
                loader = PyPDFLoader(doc_path)
                documents = loader.load() # PyPDFLoader loads page by page
                if not documents:
                    st.warning(f"No content loaded from PDF: {os.path.basename(doc_path)}")
                    print(f"DEBUG: No documents loaded from {doc_path}")
                    continue
                print(f"DEBUG: Loaded {len(documents)} pages from {doc_path}")
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200,
                    length_function=len
                )
                # Each 'document' from PyPDFLoader is a page, split each page
                chunks_from_pdf = text_splitter.split_documents(documents)
                all_document_chunks.extend(chunks_from_pdf)
                st.sidebar.write(f"Processed '{os.path.basename(doc_path)}': {len(chunks_from_pdf)} chunks total.")
                print(f"DEBUG: Split {os.path.basename(doc_path)} into {len(chunks_from_pdf)} chunks.")
            except Exception as e:
                st.sidebar.error(f"Error loading/splitting PDF '{os.path.basename(doc_path)}': {e}")
                print(f"ERROR DEBUG: Error loading/splitting {doc_path}: {e}")
        else:
            st.sidebar.warning(f"Document not found for splitting: {doc_path}")
            print(f"DEBUG: Document not found: {doc_path}")
    print(f"DEBUG: Total document chunks created from all PDFs: {len(all_document_chunks)}")
    return all_document_chunks

def initialize_vector_store(embeddings_model):
    """Initializes the vector store from the source document or loads an existing one."""
    print("DEBUG: initialize_vector_store called.")
    if embeddings_model is None:
        st.sidebar.error("Embedding model not loaded. Cannot initialize vector store.")
        print("DEBUG: Embedding model is None in initialize_vector_store.")
        return None

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    if os.path.exists(VECTOR_STORE_PATH + ".faiss") and os.path.exists(VECTOR_STORE_PATH + ".pkl"):
        try:
            print(f"DEBUG: Attempting to load existing vector store from {VECTOR_STORE_DIR}")
            vector_store = FAISS.load_local(
                folder_path=VECTOR_STORE_DIR,
                embeddings=embeddings_model, 
                index_name="faiss_index",
                allow_dangerous_deserialization=True
            )
            st.sidebar.success("Knowledge base loaded from disk.")
            print("DEBUG: Existing vector store loaded successfully.")
            if hasattr(vector_store, 'index') and vector_store.index: # Check if index attribute exists and is not None
                 print(f"DEBUG: Loaded vector store has {vector_store.index.ntotal} vectors.")
            else:
                 print("DEBUG: Loaded vector store does not have an index or index is not initialized (e.g., empty).")
            return vector_store
        except Exception as e:
            st.sidebar.warning(f"Failed to load existing knowledge base: {e}. Re-initializing...")
            print(f"ERROR DEBUG: Failed to load existing vector store: {e}")
    
    if not os.path.exists(SOURCE_DOCUMENT_PATH):
        st.sidebar.error(f"Source document '{SOURCE_DOCUMENT_FILENAME}' not found in '{DOCUMENTS_DIR}'.")
        print(f"ERROR DEBUG: Source document not found at {SOURCE_DOCUMENT_PATH}")
        return None

    st.sidebar.info(f"Creating new knowledge base from '{SOURCE_DOCUMENT_FILENAME}'...")
    print(f"DEBUG: Creating new vector store from {SOURCE_DOCUMENT_PATH}")
    document_chunks = _load_and_split_pdfs([SOURCE_DOCUMENT_PATH])
    
    if not document_chunks:
        st.sidebar.error("Failed to process the source document. No chunks created. Knowledge base not initialized.")
        print("ERROR DEBUG: No document chunks created from source PDF.")
        return None
    
    try:
        print(f"DEBUG: Creating FAISS index from {len(document_chunks)} chunks.")
        vector_store = FAISS.from_documents(document_chunks, embeddings_model)
        vector_store.save_local(folder_path=VECTOR_STORE_DIR, index_name="faiss_index")
        st.sidebar.success("New knowledge base created and saved.")
        print("DEBUG: New vector store created and saved successfully.")
        if hasattr(vector_store, 'index') and vector_store.index:
            print(f"DEBUG: New vector store has {vector_store.index.ntotal} vectors.")
        return vector_store
    except Exception as e:
        st.sidebar.error(f"Error creating vector store: {e}")
        print(f"ERROR DEBUG: Error creating vector store from documents: {e}")
        return None

def add_documents_to_vector_store(uploaded_files, vector_store, embeddings_model):
    print("DEBUG: add_documents_to_vector_store called.")
    if not uploaded_files:
        st.sidebar.info("No files were selected for upload.")
        return vector_store
    if not vector_store:
        st.sidebar.error("Knowledge base not initialized. Cannot add documents.")
        return vector_store
    if embeddings_model is None:
        st.sidebar.error("Embedding model not loaded. Cannot add documents.")
        return vector_store

    temp_doc_paths = []
    saved_files_count = 0
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    for uploaded_file in uploaded_files:
        # Create a unique name for temporary files
        temp_file_path = os.path.join(DOCUMENTS_DIR, f"uploaded_{uploaded_file.id}_{uploaded_file.name}")
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_doc_paths.append(temp_file_path)
            saved_files_count += 1
        except Exception as e:
            st.sidebar.error(f"Error saving uploaded file '{uploaded_file.name}': {e}")
            print(f"ERROR DEBUG: Error saving uploaded file {uploaded_file.name}: {e}")

    if not temp_doc_paths:
        st.sidebar.warning("No valid files could be saved for processing.")
        return vector_store

    st.sidebar.info(f"Processing {saved_files_count} uploaded document(s)...")
    new_document_chunks = _load_and_split_pdfs(temp_doc_paths) 
    
    if new_document_chunks:
        try:
            vector_store.add_documents(new_document_chunks)
            vector_store.save_local(folder_path=VECTOR_STORE_DIR, index_name="faiss_index")
            st.sidebar.success(f"{len(new_document_chunks)} new chunks added.")
            if hasattr(vector_store, 'index') and vector_store.index:
                 print(f"DEBUG: Updated vector store has {vector_store.index.ntotal} vectors.")
        except Exception as e:
            st.sidebar.error(f"Error adding documents: {e}")
            print(f"ERROR DEBUG: Error adding documents to vector store: {e}")
    else:
        st.sidebar.warning("No text could be extracted from the uploaded documents to add.")
        print("DEBUG: No new chunks extracted from uploaded documents.")

    for path in temp_doc_paths: 
        try:
            os.remove(path)
        except OSError:
            pass
            
    return vector_store


# --- LLM and RAG Chain ---
@st.cache_resource
def get_llm(api_key, model_name="gemini-1.5-flash", temperature=0.3):
    print(f"DEBUG: get_llm called with model: {model_name}, temp: {temperature}")
    if not api_key:
        st.error("Google API Key is not provided. LLM cannot be initialized.")
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            # convert_system_message_to_human=True, # REMOVED - Deprecated
        )
        print("DEBUG: LLM initialized successfully.")
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM (Model: {model_name}): {e}")
        print(f"ERROR DEBUG: Error initializing LLM: {e}")
        return None

# TEMPORARILY MODIFIED PROMPT FOR DEBUGGING - Be slightly less strict
RAG_PROMPT_TEMPLATE_STRING = """
You are an AI assistant. Your main task is to answer questions based on the provided context documents.
Please use the information from the 'Context' section below to answer the 'Question'.
If the context seems relevant, try your best to form an answer.
If the context clearly does not contain the information needed to answer the question, then state: "I'm sorry, I cannot answer this question based on the provided documents."
Do not make up information or use external knowledge beyond the provided context.

Context:
{context}

Chat History:
{chat_history}

Question: {input}

Answer:
"""
rag_prompt_template = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE_STRING)

def get_rag_chain(llm, passed_retriever):
    print("DEBUG: get_rag_chain called.")
    if llm is None or passed_retriever is None:
        st.error("LLM or Retriever not initialized. Cannot create RAG chain.")
        return None
    try:
        question_answer_chain = create_stuff_documents_chain(llm, rag_prompt_template)
        rag_chain = create_retrieval_chain(passed_retriever, question_answer_chain)
        print("DEBUG: RAG chain created successfully.")
        return rag_chain
    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        print(f"ERROR DEBUG: Error creating RAG chain: {e}")
        return None

def format_chat_history_for_prompt(streamlit_messages):
    history_string = ""
    if not streamlit_messages:
        return "No prior conversation."
    for msg in streamlit_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        text_content = ""
        if msg.get("content") and isinstance(msg["content"], list) and len(msg["content"]) > 0:
            content_item = msg["content"][0]
            if isinstance(content_item, dict) and content_item.get("type") == "text":
                text_content = content_item.get("text", "")
        history_string += f"{role}: {text_content}\n"
    return history_string.strip()
