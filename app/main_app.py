import streamlit as st
import os
import dotenv

from app.auth import init_db, login_page, register_page
from app.rag_core import (
    initialize_vector_store,
    add_documents_to_vector_store,
    get_llm,
    get_rag_chain,
    format_chat_history_for_prompt,
    get_embeddings_model
)

dotenv.load_dotenv() 

GOOGLE_MODELS_AVAILABLE = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

def chatbot_page():
    st.markdown("<h1 style='text-align: center; color: orange;'>DocuMentor RAG</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Chat with your PDF documents intelligently!</p>", unsafe_allow_html=True)

    # Initialize session state flags if they don't exist
    if "google_api_key" not in st.session_state:
        st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY", "")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Flags for auto-loading process
    if "embeddings_model_loaded" not in st.session_state:
        st.session_state.embeddings_model_loaded = False
    if "embeddings_model" not in st.session_state: # Stores the actual model object
        st.session_state.embeddings_model = None
    if "embedding_model_load_attempted" not in st.session_state: # To try auto-load once
        st.session_state.embedding_model_load_attempted = False

    if "vector_store_initialized" not in st.session_state:
        st.session_state.vector_store_initialized = False
    if "vector_store" not in st.session_state: # Stores the actual vector store object
        st.session_state.vector_store = None
    if "vector_store_init_attempted" not in st.session_state: # To try auto-init once
        st.session_state.vector_store_init_attempted = False

    # --- Auto-Initialization Phase ---
    # 1. Load Embedding Model (if not already loaded and not yet attempted automatically)
    if not st.session_state.embeddings_model_loaded and not st.session_state.embedding_model_load_attempted:
        st.session_state.embedding_model_load_attempted = True # Mark as attempted for this session run
        with st.spinner("Loading embedding model... Please wait."):
            print("DEBUG Main_App: Auto-loading embedding model...")
            st.session_state.embeddings_model = get_embeddings_model()
            if st.session_state.embeddings_model:
                st.session_state.embeddings_model_loaded = True
                print("DEBUG Main_App: Embedding model auto-loaded successfully.")
                st.rerun() # Rerun to reflect state change and proceed to KB init if needed
            else:
                print("DEBUG Main_App: Embedding model auto-load FAILED.")
                # Sidebar will show warning and allow manual retry

    # 2. Initialize Vector Store (if embeddings loaded, KB not yet initialized, and not yet attempted automatically)
    if st.session_state.embeddings_model_loaded and \
       not st.session_state.vector_store_initialized and \
       not st.session_state.vector_store_init_attempted:
        st.session_state.vector_store_init_attempted = True # Mark as attempted
        with st.spinner("Initializing Knowledge Base... This may take a few moments."):
            print("DEBUG Main_App: Auto-initializing vector store...")
            st.session_state.vector_store = initialize_vector_store(st.session_state.embeddings_model)
            if st.session_state.vector_store:
                st.session_state.vector_store_initialized = True
                print("DEBUG Main_App: Vector store auto-initialized successfully.")
                st.rerun() # Rerun to reflect state change
            else:
                print("DEBUG Main_App: Vector store auto-initialization FAILED.")
                # Sidebar will show warning and allow manual retry


    # --- Sidebar ---
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.get('username', 'Guest')}!")
        if st.button("Logout", key="logout_btn_main_app_auto"):
            keys_to_clear = ["logged_in", "username", "messages", 
                             "vector_store_initialized", "vector_store",
                             "embeddings_model_loaded", "embeddings_model",
                             "embedding_model_load_attempted", "vector_store_init_attempted"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state["page"] = "login"
            st.rerun()
        st.divider()

        st.subheader("Configuration")
        current_api_key = st.session_state.google_api_key
        google_api_key_input = st.text_input(
            "Google API Key:", value=current_api_key, type="password",
            key="google_api_key_input_ui_auto", help="Enter your Google API Key."
        )
        if google_api_key_input != current_api_key:
             st.session_state.google_api_key = google_api_key_input
        
        selected_model = st.selectbox(
            "Select LLM Model:", GOOGLE_MODELS_AVAILABLE, index=0, key="llm_model_select_ui_auto"
        )
        temperature = st.slider(
            "Temperature (Creativity):", 0.0, 1.0, 0.3, 0.05, key="temperature_slider_ui_auto"
        )
        st.divider()

        st.subheader("Knowledge Base Status")
        if st.session_state.embeddings_model_loaded:
            st.sidebar.success("✅ Embedding Model Loaded")
            
            if st.session_state.vector_store_initialized and st.session_state.vector_store:
                st.sidebar.success("✅ Knowledge Base Ready")
                uploaded_files = st.file_uploader(
                    "Upload more PDF documents to add:", type=["pdf"],
                    accept_multiple_files=True, key="pdf_uploader_ui_sidebar_auto"
                )
                if uploaded_files:
                    # Ensure embeddings model is still available (it should be due to @st.cache_resource)
                    if st.session_state.embeddings_model:
                        with st.spinner("Processing new documents..."):
                            st.session_state.vector_store = add_documents_to_vector_store(
                                uploaded_files, st.session_state.vector_store, st.session_state.embeddings_model
                            )
                        st.rerun()
                    else:
                        st.sidebar.error("Embedding model not available for adding documents. Please reload.")
            
            elif st.session_state.vector_store_init_attempted and not st.session_state.vector_store:
                 st.sidebar.error("⚠️ KB auto-initialization failed.")
                 if st.button("Retry Knowledge Base Init", key="retry_kb_init_manual_auto"):
                    st.session_state.vector_store_init_attempted = False # Allow another auto-attempt cycle
                    st.rerun()
            else: # Embeddings loaded, KB not initialized, and not yet attempted (or attempt flag reset)
                st.sidebar.info("🔄 Knowledge Base initializing automatically...")
                # Auto-init logic at the top of the page handles this.
                # Manual button if user wants to force it or if auto-init seems stuck (though auto should trigger)
                if st.button("Initialize/Load Knowledge Base (Manual)", key="init_kb_btn_manual_auto"):
                    st.session_state.vector_store_init_attempted = False # Reset attempt flag for this manual try
                    st.rerun()


        else: # Embedding model not loaded
            st.sidebar.warning("⚠️ Embedding Model Not Loaded")
            if st.session_state.embedding_model_load_attempted and not st.session_state.embeddings_model:
                st.sidebar.error("Embedding model auto-load failed.")
            if st.button("Load/Retry Embedding Model", key="retry_emb_load_manual_auto"):
                st.session_state.embedding_model_load_attempted = False # Allow another auto-attempt cycle
                st.rerun()
        
        st.divider()
        if st.button("🗑️ Reset Chat History", key="reset_chat_btn_ui_auto"):
            st.session_state.messages = []
            st.rerun()

    # --- Main Chat Interface ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message.get("content"), list) and len(message["content"]) > 0:
                 if message["content"][0].get("type") == "text":
                    st.markdown(message["content"][0].get("text", ""))

    prompt_disabled = False
    disabled_reason = ""
    if not st.session_state.google_api_key:
        prompt_disabled = True
        disabled_reason += "Google API Key missing. "
    if not st.session_state.embeddings_model_loaded:
        prompt_disabled = True
        disabled_reason += "Embedding model not loaded. "
    if not st.session_state.vector_store_initialized or not st.session_state.vector_store:
        prompt_disabled = True
        disabled_reason += "Knowledge Base not initialized. "

    if prompt_disabled:
        # Show a general waiting message if auto-initialization might still be in progress
        if not st.session_state.embeddings_model_loaded or \
           (st.session_state.embeddings_model_loaded and not st.session_state.vector_store_initialized and \
            not (st.session_state.vector_store_init_attempted and not st.session_state.vector_store) ): # not (attempted AND failed)
            st.info("🔄 Setting up chatbot essentials (Embedding Model & Knowledge Base)... Please wait or check sidebar.")
        elif disabled_reason: # If setup is done or failed, show specific reasons
             st.info(f"⬅️ Chat disabled. Please resolve in sidebar: {disabled_reason.strip()}")


    if prompt := st.chat_input("Ask a question about the documents...", disabled=prompt_disabled, key="chat_input_main_auto"):
        print(f"\nDEBUG: User prompt: {prompt}")
        # Re-check critical prerequisites before processing
        if not st.session_state.google_api_key or \
           not st.session_state.embeddings_model_loaded or \
           not st.session_state.vector_store_initialized or \
           not st.session_state.vector_store:
            
            error_msg = "Cannot process request: "
            if not st.session_state.google_api_key: error_msg += "Google API Key missing. "
            if not st.session_state.embeddings_model_loaded: error_msg += "Embedding model not loaded. "
            if not st.session_state.vector_store_initialized or not st.session_state.vector_store: error_msg += "Knowledge Base not initialized. "
            st.error(f"⬅️ {error_msg.strip()} Please check sidebar configuration.")
            st.stop() # Critical failure, stop processing this prompt

        st.session_state.messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        with st.chat_message("user"):
            st.markdown(prompt)

        llm = get_llm(api_key=st.session_state.google_api_key, model_name=selected_model, temperature=temperature)
        if not llm:
            st.session_state.messages.append({"role": "assistant", "content": [{"type": "text", "text": "Error: LLM not initialized. Check API key."}]})
            st.rerun()
            
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        
        print("DEBUG: Attempting to retrieve documents with retriever...")
        try:
            retrieved_docs = retriever.invoke(prompt)
            print(f"DEBUG: Retriever found {len(retrieved_docs)} documents.")
            if not retrieved_docs: print("DEBUG: Retriever found NO documents for the prompt.")
            for i, doc in enumerate(retrieved_docs):
                print(f"--- Retrieved Doc {i+1} (Metadata: {doc.metadata}) ---")
                print(f"Content (first 150 chars): {doc.page_content[:150]}...")
        except Exception as e_retrieve:
            print(f"ERROR DEBUG: Error during retriever.invoke: {e_retrieve}")
            retrieved_docs = []

        rag_chain = get_rag_chain(llm, retriever)
        if not rag_chain:
            st.session_state.messages.append({"role": "assistant", "content": [{"type": "text", "text": "Error: RAG chain not initialized."}]})
            st.rerun()

        if llm and rag_chain: # Should be true if we haven't reran due to errors
            chat_history_for_prompt = format_chat_history_for_prompt(st.session_state.messages[:-1])
            print(f"DEBUG: Chat history for prompt: '{chat_history_for_prompt[:100]}...'")
            # print(f"DEBUG: Context for RAG chain (from retrieved_docs): {[doc.page_content[:100] for doc in retrieved_docs]}")

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response_content = ""
                try:
                    response_payload = {"input": prompt, "chat_history": chat_history_for_prompt}
                    print(f"DEBUG: Invoking RAG chain with payload: {{'input': '{prompt[:50]}...', 'chat_history': '{chat_history_for_prompt[:50]}...'}}")
                    
                    for chunk in rag_chain.stream(response_payload):
                        if "answer" in chunk: 
                            full_response_content += chunk["answer"]
                            message_placeholder.markdown(full_response_content + "▌")
                    message_placeholder.markdown(full_response_content)
                    print(f"DEBUG: RAG chain full response: {full_response_content}")

                except Exception as e_rag:
                    st.error(f"Error during RAG chain execution: {e_rag}")
                    print(f"ERROR DEBUG: RAG chain execution error: {e_rag}")
                    full_response_content = "Sorry, an error occurred processing your request."
                    message_placeholder.markdown(full_response_content)
                
                st.session_state.messages.append({"role": "assistant", "content": [{"type": "text", "text": full_response_content}]})

def main():
    init_db()
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    if st.session_state["logged_in"]:
        st.set_page_config(page_title="DocuMentor RAG", page_icon="📚", layout="wide", initial_sidebar_state="expanded")
        chatbot_page()
    elif st.session_state["page"] == "login":
        st.set_page_config(page_title="Login - DocuMentor RAG", layout="centered", initial_sidebar_state="auto")
        login_page()
        if st.button("Go to Register", key="goto_register_btn_login_auto"): # Unique key
            st.session_state["page"] = "register"
            st.rerun()
    elif st.session_state["page"] == "register":
        st.set_page_config(page_title="Register - DocuMentor RAG", layout="centered", initial_sidebar_state="auto")
        register_page()
        if st.button("Go to Login", key="goto_login_btn_register_auto"): # Unique key
            st.session_state["page"] = "login"
            st.rerun()
    else: 
        st.session_state["page"] = "login"
        st.rerun()

if __name__ == "__main__":
    main()
