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

# --- Agent Imports ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool
from langchain import hub # To pull pre-defined prompts
from app.config import TAVILY_API_KEY # Import Tavily API key

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
    print(f"DEBUG: _load_and_split_pdfs called with {len(pdf_paths)} paths")
    for doc_path in pdf_paths:
        if os.path.exists(doc_path):
            try:
                # Don't print every file path to reduce log spam
                loader = PyPDFLoader(doc_path)
                documents = loader.load() # PyPDFLoader loads page by page
                if not documents:
                    st.warning(f"No content loaded from PDF: {os.path.basename(doc_path)}")
                    print(f"DEBUG: No documents loaded from {os.path.basename(doc_path)}")
                    continue
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200,
                    length_function=len
                )
                # Each 'document' from PyPDFLoader is a page, split each page
                chunks_from_pdf = text_splitter.split_documents(documents)
                all_document_chunks.extend(chunks_from_pdf)
                
                # Print summary instead of each document details
                st.sidebar.write(f"Processed '{os.path.basename(doc_path)}': {len(chunks_from_pdf)} chunks total.")
            except Exception as e:
                st.sidebar.error(f"Error loading/splitting PDF '{os.path.basename(doc_path)}': {e}")
                print(f"ERROR DEBUG: Error loading/splitting {os.path.basename(doc_path)}: {e}")
        else:
            st.sidebar.warning(f"Document not found for splitting: {os.path.basename(doc_path)}")
            print(f"DEBUG: Document not found: {os.path.basename(doc_path)}")
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

def get_all_document_names():
    """Returns a list of all document names in the documents directory."""
    import glob
    document_files = glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf"))
    return [os.path.basename(doc) for doc in document_files]

def resync_knowledge_base(embeddings_model):
    """Rebuilds the entire knowledge base by processing all documents in the documents directory."""
    if embeddings_model is None:
        st.sidebar.error("Embedding model not loaded. Cannot sync knowledge base.")
        return None
    
    import glob
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    # Get all PDF files in the documents directory
    all_document_paths = glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf"))
    
    if not all_document_paths:
        st.sidebar.warning("No documents found to sync in the knowledge base.")
        return None
    
    st.sidebar.info(f"Syncing knowledge base with {len(all_document_paths)} documents...")
    print(f"DEBUG: Rebuilding vector store with {len(all_document_paths)} documents")
    
    # Process all documents at once
    all_document_chunks = _load_and_split_pdfs(all_document_paths)
    
    if not all_document_chunks:
        st.sidebar.error("Failed to process documents. No chunks created. Knowledge base not updated.")
        return None
    
    try:
        vector_store = FAISS.from_documents(all_document_chunks, embeddings_model)
        vector_store.save_local(folder_path=VECTOR_STORE_DIR, index_name="faiss_index")
        st.sidebar.success(f"Knowledge base synced with {len(all_document_chunks)} chunks from {len(all_document_paths)} documents.")
        if hasattr(vector_store, 'index') and vector_store.index:
            print(f"DEBUG: Synced vector store has {vector_store.index.ntotal} vectors")
        return vector_store
    except Exception as e:
        st.sidebar.error(f"Error syncing knowledge base: {e}")
        print(f"ERROR: Failed to create vector store during sync: {e}")
        return None

def remove_document_from_kb(document_name, vector_store, embeddings_model):
    """Removes a document from the knowledge base by rebuilding the vector store without it."""
    if embeddings_model is None or vector_store is None:
        st.sidebar.error("Embedding model or vector store not available. Cannot remove document.")
        return vector_store
    
    import glob
    document_path = os.path.join(DOCUMENTS_DIR, document_name)
    
    # First check if the document exists
    if not os.path.exists(document_path):
        st.sidebar.error(f"Document '{document_name}' not found.")
        return vector_store
    
    # Get all documents except the one to remove
    all_document_paths = glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf"))
    remaining_documents = [doc for doc in all_document_paths if os.path.basename(doc) != document_name]
    
    if not remaining_documents:
        st.sidebar.warning("This is the only document in the knowledge base. Removing it will empty the KB.")
        try:
            # Create an empty vector store
            empty_vector_store = FAISS.from_documents([], embeddings_model)
            empty_vector_store.save_local(folder_path=VECTOR_STORE_DIR, index_name="faiss_index")
            # Delete the document file from the documents folder
            try:
                os.remove(document_path)
                st.sidebar.success(f"Document '{document_name}' removed from knowledge base and disk.")
            except Exception as e:
                st.sidebar.warning(f"Document removed from KB but could not delete file: {e}")
            return empty_vector_store
        except Exception as e:
            st.sidebar.error(f"Error creating empty vector store: {e}")
            return vector_store
    
    # Rebuild vector store with remaining documents
    st.sidebar.info(f"Rebuilding knowledge base without '{document_name}'...")
    all_document_chunks = _load_and_split_pdfs(remaining_documents)
    
    if not all_document_chunks:
        st.sidebar.error("Failed to process remaining documents. Knowledge base not updated.")
        return vector_store
    
    try:
        new_vector_store = FAISS.from_documents(all_document_chunks, embeddings_model)
        new_vector_store.save_local(folder_path=VECTOR_STORE_DIR, index_name="faiss_index")
        # Delete the document file from the documents folder
        try:
            os.remove(document_path)
            st.sidebar.success(f"Document '{document_name}' removed from knowledge base and disk.")
        except Exception as e:
            st.sidebar.warning(f"Document removed from KB but could not delete file: {e}")
        return new_vector_store
    except Exception as e:
        st.sidebar.error(f"Error rebuilding vector store: {e}")
        return vector_store

def add_documents_to_vector_store(uploaded_files, vector_store, embeddings_model):
    print(f"DEBUG: Processing {len(uploaded_files)} uploaded files")
    if not uploaded_files:
        st.sidebar.info("No files were selected for upload.")
        return vector_store
    if not vector_store:
        st.sidebar.error("Knowledge base not initialized. Cannot add documents.")
        return vector_store
    if embeddings_model is None:
        st.sidebar.error("Embedding model not loaded. Cannot add documents.")
        return vector_store

    import datetime
    
    doc_paths = []
    saved_files_count = 0
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    for uploaded_file in uploaded_files:
        # Generate timestamp for uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Get file name parts
        file_name, file_ext = os.path.splitext(uploaded_file.name)
        # Create a name for permanent storage in documents folder with timestamp to avoid overwriting
        file_path = os.path.join(DOCUMENTS_DIR, f"{file_name}_{timestamp}{file_ext}")
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            doc_paths.append(file_path)
            saved_files_count += 1
        except Exception as e:
            st.sidebar.error(f"Error saving uploaded file '{uploaded_file.name}': {e}")
            print(f"ERROR: Failed to save '{uploaded_file.name}': {e}")

    if not doc_paths:
        st.sidebar.warning("No valid files could be saved for processing.")
        return vector_store

    st.sidebar.info(f"Processing {saved_files_count} uploaded document(s)...")
    new_document_chunks = _load_and_split_pdfs(doc_paths) 
    
    if new_document_chunks:
        try:
            vector_store.add_documents(new_document_chunks)
            vector_store.save_local(folder_path=VECTOR_STORE_DIR, index_name="faiss_index")
            st.sidebar.success(f"{len(new_document_chunks)} new chunks added. Files stored in documents folder.")
            if hasattr(vector_store, 'index') and vector_store.index:
                 print(f"DEBUG: Vector store updated with {len(new_document_chunks)} new chunks. Total: {vector_store.index.ntotal} vectors.")
        except Exception as e:
            st.sidebar.error(f"Error adding documents: {e}")
            print(f"ERROR: Failed to add documents to vector store: {e}")
    else:
        st.sidebar.warning("No text could be extracted from the uploaded documents to add.")
        print("DEBUG: No chunks created from uploaded documents")
            
    return vector_store


# --- LLM and RAG Chain ---
@st.cache_resource
def get_llm(api_key, model_name="gemini-1.5-flash", temperature=0.3):
    print(f"DEBUG: get_llm called with model: {model_name}, temp: {temperature}")
    if not api_key:
        print("ERROR: Google API Key is not provided. LLM cannot be initialized in get_llm.")
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
        print(f"ERROR DEBUG: Error initializing LLM (Model: {model_name}): {e}")
        return None

# TEMPORARILY MODIFIED PROMPT FOR DEBUGGING - Be slightly less strict
RAG_PROMPT_TEMPLATE_STRING = """
You are a knowledgeable AI assistant specializing in document analysis and information retrieval.

INSTRUCTIONS:
1. When answering questions, prioritize information from the provided context documents
2. Always cite your sources by mentioning which document or section contains the information
3. If the context doesn't contain relevant information, clearly state this and then use your general knowledge
4. For multi-part questions, address each part systematically
5. When summarizing documents, include key points, main findings, and significant details
6. If information in the context seems contradictory, acknowledge this and explain possible interpretations
7. Maintain a professional, objective tone in your responses
8. If information is time-sensitive, note when the source documents were created
9. For numerical data from documents, present it accurately with proper context

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
        print("ERROR: LLM or Retriever not initialized. Cannot create RAG chain.")
        return None
    try:
        question_answer_chain = create_stuff_documents_chain(llm, rag_prompt_template)
        rag_chain = create_retrieval_chain(passed_retriever, question_answer_chain)
        print("DEBUG: RAG chain created successfully.")
        return rag_chain
    except Exception as e:
        print(f"ERROR DEBUG: Error creating RAG chain: {e}")
        return None

def format_chat_history_for_prompt(streamlit_messages):
    """Converts Streamlit message history to a list of LangChain BaseMessage objects."""
    history_messages = []
    if not streamlit_messages:
        return [] # Return empty list for agent
    for msg in streamlit_messages:
        role = msg.get("role")
        content = msg.get("content")
        
        # Streamlit often stores content as a list of dicts e.g., [{"type": "text", "text": "..."}]
        # Or sometimes directly as a string if it's simple.
        text_content = ""
        if isinstance(content, list) and len(content) > 0:
            # Assuming the first item in the list is the one we want if it's complex
            content_item = content[0]
            if isinstance(content_item, dict) and content_item.get("type") == "text":
                text_content = content_item.get("text", "")
            elif isinstance(content_item, str): # If content is like ["text"]
                text_content = content_item
        elif isinstance(content, str):
            text_content = content
        else:
            # Fallback for unexpected content structure
            print(f"WARN: Unexpected message content structure: {content}")
            text_content = str(content)

        if role == "user":
            history_messages.append(HumanMessage(content=text_content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=text_content))
        # Silently ignore system messages or other roles for now, or handle as needed
            
    # print(f"DEBUG: Formatted chat history for agent: {history_messages}")
    return history_messages

def get_knowledge_base_stats():
    """Returns statistics about the knowledge base, including document count and vector count."""
    stats = {
        "document_count": 0,
        "vector_count": 0,
        "vector_store_exists": False,
        "documents_exist": False
    }
    
    # Check if vector store files exist
    if os.path.exists(VECTOR_STORE_PATH + ".faiss") and os.path.exists(VECTOR_STORE_PATH + ".pkl"):
        stats["vector_store_exists"] = True
    
    # Count documents
    import glob
    document_files = glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf"))
    stats["document_count"] = len(document_files)
    stats["documents_exist"] = len(document_files) > 0
    
    # Try to get vector count
    try:
        if stats["vector_store_exists"]:
            # FAISS doesn't provide a way to get vector count without loading
            # This is just a placeholder - in a real app we might store this metadata separately
            stats["vector_count"] = "Available after loading vector store"
    except Exception:
        pass
    
    # Get total size of vector store files
    vector_store_size = 0
    if os.path.exists(VECTOR_STORE_PATH + ".faiss"):
        vector_store_size += os.path.getsize(VECTOR_STORE_PATH + ".faiss")
    if os.path.exists(VECTOR_STORE_PATH + ".pkl"):
        vector_store_size += os.path.getsize(VECTOR_STORE_PATH + ".pkl")
    
    stats["vector_store_size_mb"] = round(vector_store_size / (1024 * 1024), 2) if vector_store_size > 0 else 0
    
    return stats

# --- Agent and Tools ---

def get_tools(llm, retriever):
    """Defines the tools available to the agent."""
    print("DEBUG: get_tools called.")
    if llm is None:
        print("ERROR: LLM is None in get_tools. Cannot create RAG tool.")
        # Potentially return a list without the RAG tool or handle this error upstream
        # For now, let's allow it to proceed and Tavily tool can still be created.
        # return [] 
    
    # 1. RAG Tool (Document Search)
    # We need to define a function that the RAG tool can call.
    # This function will invoke the existing RAG chain.
    
    _rag_chain_for_tool = None
    if llm and retriever:
        try:
            # This is the runnable that expects {'input': ..., 'chat_history': ...}
            # and internally handles retrieval and generation.
            question_answer_chain = create_stuff_documents_chain(llm, rag_prompt_template)
            _rag_chain_for_tool = create_retrieval_chain(retriever, question_answer_chain)
            print("DEBUG: Runnable for RAG tool created successfully.")
        except Exception as e:
            print(f"ERROR: Could not create runnable for RAG tool: {e}")
            # st.warning(f"Could not create RAG tool: {e}") # Avoid Streamlit calls here

    def invoke_rag_tool(query: str):
        """
        Invokes the RAG chain to answer questions based on available documents.
        Use this tool for questions about specific information that might be contained
        within the loaded PDF documents. For example, 'What were the company's revenues last year?'
        or 'Summarize the risk factors section of the document.'
        The user's query will be passed as input. Chat history will be retrieved from the session.
        """
        if not _rag_chain_for_tool:
            return "Error: RAG chain (document search) is not available."
        
        # Retrieve chat history from Streamlit session state
        # This assumes 'messages' is the key where Streamlit app stores chat history
        # and format_chat_history_for_prompt can convert it.
        chat_history_for_rag = []
        if "messages" in st.session_state and st.session_state.messages:
            # Convert Streamlit messages to LangChain BaseMessage objects if needed by the RAG chain
            # The create_retrieval_chain expects a list of BaseMessages (HumanMessage, AIMessage)
            raw_history = st.session_state.messages
            for msg in raw_history:
                if msg["role"] == "user":
                    chat_history_for_rag.append(HumanMessage(content=msg["content"])) # Streamlit usually stores content directly
                elif msg["role"] == "assistant":
                    chat_history_for_rag.append(AIMessage(content=msg["content"])) # Streamlit usually stores content directly
            print(f"DEBUG: Retrieved chat history from st.session_state for RAG tool: {len(chat_history_for_rag)} messages")
        else:
            print("DEBUG: No chat history found in st.session_state.messages for RAG tool.")

        try:
            print(f"DEBUG: RAG Tool invoking with query: '{query}' and history_len: {len(chat_history_for_rag)}")
            result = _rag_chain_for_tool.invoke({
                "input": query,
                "chat_history": chat_history_for_rag 
            })
            answer = result.get("answer", "No answer found by RAG.")
            print(f"DEBUG: RAG Tool received answer: '{answer}'")
            return answer
        except Exception as e:
            print(f"ERROR: Error invoking RAG tool: {e}")
            return f"Error during RAG tool execution: {e}"

    rag_document_tool = Tool(
        name="DocumentSearch",
        func=invoke_rag_tool, 
        description="Searches and retrieves information from the loaded PDF documents. Use this for specific questions about the content of these documents, like financial data, company policies, technical specifications, or summaries of document sections. Input should be the user's question. This tool automatically uses the conversation history for context."
    )

    # 2. Tavily Search Tool (Internet Search)
    tavily_tool = TavilySearchResults(
        max_results=3,
        api_key=TAVILY_API_KEY # Ensure TAVILY_API_KEY is set in your environment or config
    )
    tavily_tool.description = "A search engine for finding information on the internet. Use this when the user asks for information not found in the documents, or for current events, general knowledge, or public data. Input should be a search query."


    # 3. Weather Tool (Placeholder)
    def get_current_weather(location: str, unit: str = "celsius"):
        """
        Gets the current weather for a specified location.
        Use this tool when the user asks about the weather.
        Specify the location and optionally the unit (celsius or fahrenheit).
        """
        # This is a placeholder. In a real scenario, you would call a weather API.
        print(f"DEBUG: Weather tool called for location: {location}, unit: {unit}")
        if "tokyo" in location.lower():
            return f"The current weather in Tokyo is 25°{unit.upper()[0]} and sunny."
        elif "london" in location.lower():
            return f"The current weather in London is 15°{unit.upper()[0]} and cloudy."
        else:
            return f"Sorry, I don't have the weather information for {location} right now."

    weather_tool = Tool(
        name="GetWeather",
        func=get_current_weather, # Langchain will inspect the function signature for args
        description="Provides the current weather for a given location. Input should be the location (e.g., 'Tokyo'). Optional unit can be 'celsius' or 'fahrenheit'."
    )
    
    tools = [rag_document_tool, tavily_tool, weather_tool]
    print(f"DEBUG: Tools created: {[tool.name for tool in tools]}")
    return tools


@st.cache_resource(experimental_allow_widgets=True) # Allow widgets for potential future tool interactions
def get_agent_executor(_llm, _retriever, google_api_key, tavily_api_key_val):
    """Creates and returns a ReAct agent executor."""
    print("DEBUG: get_agent_executor called.")
    if not _llm:
        # st.error("LLM not initialized. Cannot create agent.")
        print("ERROR: LLM not initialized in get_agent_executor. Cannot create agent.")
        return None
    # Retriever can be None if only web search or other tools are desired initially.
    if not _retriever:
        print("WARNING: Retriever not initialized in get_agent_executor. RAG tool might not function fully or at all.")

    if not tavily_api_key_val:
        print("WARNING: TAVILY_API_KEY not found. TavilySearch tool will not work.")

    tools = get_tools(_llm, _retriever) # Pass the underscore-prefixed llm and retriever
    
    try:
        prompt = hub.pull("hwchase17/react-chat")
        print(f"DEBUG: Agent prompt pulled successfully: {prompt}")
    except Exception as e:
        print(f"ERROR: Failed to pull agent prompt from LangChain Hub: {e}. Using a basic fallback.")
        return None

    try:
        agent = create_react_agent(_llm, tools, prompt) # Use the underscore-prefixed llm
        print("DEBUG: ReAct agent created successfully.")
    except Exception as e:
        print(f"ERROR: Failed to create ReAct agent: {e}")
        return None

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  
        handle_parsing_errors=True, 
    )
    print("DEBUG: AgentExecutor created successfully.")
    return agent_executor
