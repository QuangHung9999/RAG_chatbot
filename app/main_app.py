import streamlit as st
import os
import dotenv

# Assuming auth.py, rag_core.py are in the 'app' directory
# Ensure runner.py correctly sets up sys.path for these imports
from app.auth import init_db, login_page, register_page
from app.rag_core import (
    initialize_vector_store,
    add_documents_to_vector_store,
    get_llm,
    get_rag_chain,
    format_chat_history_for_prompt,
    get_embeddings_model,
)

dotenv.load_dotenv()

GOOGLE_MODELS_AVAILABLE = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]


def chatbot_page():
    """Main page for the RAG chatbot interface."""
    st.markdown(
        "<h1 style='text-align: center; color: orange;'>DocuMentor RAG</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center;'>Chat with your PDF documents intelligently!</p>",
        unsafe_allow_html=True,
    )

    # Initialize session state flags if they don't exist
    default_keys = {
        "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
        "messages": [],
        "embeddings_model_loaded": False,
        "embeddings_model": None,
        "embedding_model_load_attempted": False,
        "vector_store_initialized": False,
        "vector_store": None,
        "vector_store_init_attempted": False,
    }
    for key, value in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # --- Auto-Initialization Phase ---
    # 1. Load Embedding Model
    if (
        not st.session_state.embeddings_model_loaded
        and not st.session_state.embedding_model_load_attempted
    ):
        st.session_state.embedding_model_load_attempted = True
        with st.spinner("Loading embedding model... Please wait."):
            print("DEBUG main_app: Auto-loading embedding model...")
            st.session_state.embeddings_model = get_embeddings_model()
            if st.session_state.embeddings_model:
                st.session_state.embeddings_model_loaded = True
                print("DEBUG main_app: Embedding model auto-loaded successfully.")
                st.rerun()
            else:
                print("DEBUG main_app: Embedding model auto-load FAILED.")

    # 2. Initialize Vector Store
    if (
        st.session_state.embeddings_model_loaded
        and not st.session_state.vector_store_initialized
        and not st.session_state.vector_store_init_attempted
    ):
        st.session_state.vector_store_init_attempted = True
        with st.spinner("Initializing Knowledge Base... This may take a few moments."):
            print("DEBUG main_app: Auto-initializing vector store...")
            st.session_state.vector_store = initialize_vector_store(
                st.session_state.embeddings_model
            )
            if st.session_state.vector_store:
                st.session_state.vector_store_initialized = True
                print("DEBUG main_app: Vector store auto-initialized successfully.")
                st.rerun()
            else:
                print("DEBUG main_app: Vector store auto-initialization FAILED.")

    # --- Sidebar ---
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.get('username', 'Guest')}!")
        if st.button("Logout", key="logout_btn_main_app_streaming"):
            keys_to_clear = list(default_keys.keys()) + [
                "logged_in",
                "username",
                "page",
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state["page"] = "login"
            st.rerun()
        st.divider()

        st.subheader("Configuration")
        current_api_key = st.session_state.google_api_key
        google_api_key_input = st.text_input(
            "Google API Key:",
            value=current_api_key,
            type="password",
            key="google_api_key_input_ui_streaming",
            help="Enter your Google API Key.",
        )
        if google_api_key_input != current_api_key:
            st.session_state.google_api_key = google_api_key_input

        selected_model = st.selectbox(
            "Select LLM Model:",
            GOOGLE_MODELS_AVAILABLE,
            index=0,
            key="llm_model_select_ui_streaming",
        )
        temperature = st.slider(
            "Temperature (Creativity):",
            0.0,
            1.0,
            0.3,
            0.05,
            key="temperature_slider_ui_streaming",
        )
        st.divider()

        st.subheader("Knowledge Base Status")
        if st.session_state.embeddings_model_loaded:
            st.sidebar.success("✅ Embedding Model Loaded")
            if (
                st.session_state.vector_store_initialized
                and st.session_state.vector_store
            ):
                st.sidebar.success("✅ Knowledge Base Ready")
                uploaded_files = st.file_uploader(
                    "Upload more PDF documents to add:",
                    type=["pdf"],
                    accept_multiple_files=True,
                    key="pdf_uploader_ui_sidebar_streaming",
                )
                if uploaded_files:
                    if st.session_state.embeddings_model:
                        with st.spinner("Processing new documents..."):
                            st.session_state.vector_store = (
                                add_documents_to_vector_store(
                                    uploaded_files,
                                    st.session_state.vector_store,
                                    st.session_state.embeddings_model,
                                )
                            )
                        st.rerun()
                    else:
                        st.sidebar.error(
                            "Embedding model not available for adding documents. Please retry loading."
                        )

            elif (
                st.session_state.vector_store_init_attempted
                and not st.session_state.vector_store
            ):
                st.sidebar.error("⚠️ KB auto-initialization failed.")
                if st.button(
                    "Retry Knowledge Base Init", key="retry_kb_init_manual_streaming"
                ):
                    st.session_state.vector_store_init_attempted = False
                    st.rerun()
            else:
                st.sidebar.info("🔄 Knowledge Base initializing...")
                if st.button(
                    "Initialize/Load Knowledge Base (Manual)",
                    key="init_kb_btn_manual_streaming",
                ):
                    st.session_state.vector_store_init_attempted = False
                    st.rerun()
        else:
            st.sidebar.warning("⚠️ Embedding Model Not Loaded")
            if (
                st.session_state.embedding_model_load_attempted
                and not st.session_state.embeddings_model
            ):
                st.sidebar.error("Embedding model auto-load failed.")
            if st.button(
                "Load/Retry Embedding Model", key="retry_emb_load_manual_streaming"
            ):
                st.session_state.embedding_model_load_attempted = False
                st.rerun()

        st.divider()
        if st.button("🗑️ Reset Chat History", key="reset_chat_btn_ui_streaming"):
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
    if (
        not st.session_state.vector_store_initialized
        or not st.session_state.vector_store
    ):
        prompt_disabled = True
        disabled_reason += "Knowledge Base not initialized. "

    if prompt_disabled:
        if not st.session_state.embeddings_model_loaded or (
            st.session_state.embeddings_model_loaded
            and not st.session_state.vector_store_initialized
            and not (
                st.session_state.vector_store_init_attempted
                and not st.session_state.vector_store
            )
        ):
            st.info("🔄 Setting up chatbot essentials... Please wait or check sidebar.")
        elif disabled_reason:
            st.info(f"⬅️ Chat disabled. Resolve in sidebar: {disabled_reason.strip()}")

    if prompt := st.chat_input(
        "Ask a question about the documents...",
        disabled=prompt_disabled,
        key="chat_input_main_streaming",
    ):
        print(f"\nDEBUG main_app: User prompt: {prompt}")
        critical_failure = False
        error_msg_display = "Cannot process request: "
        if not st.session_state.google_api_key:
            error_msg_display += "Google API Key missing. "
            critical_failure = True
        if not st.session_state.embeddings_model_loaded:
            error_msg_display += "Embedding model not loaded. "
            critical_failure = True
        if (
            not st.session_state.vector_store_initialized
            or not st.session_state.vector_store
        ):
            error_msg_display += "Knowledge Base not initialized. "
            critical_failure = True

        if critical_failure:
            st.error(
                f"⬅️ {error_msg_display.strip()}Please check sidebar configuration."
            )
            st.stop()

        st.session_state.messages.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        llm = get_llm(
            api_key=st.session_state.google_api_key,
            model_name=selected_model,
            temperature=temperature,
        )
        if not llm:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Error: LLM could not be initialized. Check API key.",
                        }
                    ],
                }
            )
            st.rerun()

        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 10}
        )

        print("DEBUG main_app: Attempting to retrieve documents with retriever...")
        try:
            retrieved_docs = retriever.invoke(prompt)
            # (Debug printing for retrieved_docs can be kept here if needed)
        except Exception as e_retrieve:
            print(f"ERROR main_app: Error during retriever.invoke: {e_retrieve}")
            retrieved_docs = []

        rag_chain = get_rag_chain(llm, retriever)
        if not rag_chain:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Error: RAG chain could not be initialized.",
                        }
                    ],
                }
            )
            st.rerun()

        if llm and rag_chain:
            chat_history_for_prompt = format_chat_history_for_prompt(
                st.session_state.messages[:-1]
            )

            with st.chat_message("assistant"):
                try:
                    response_payload = {
                        "input": prompt,
                        "chat_history": chat_history_for_prompt,
                    }
                    print(
                        f"DEBUG main_app: Invoking RAG chain with payload for streaming..."
                    )

                    # --- STREAMING IMPLEMENTATION ---
                    # Define a generator function that yields content from the stream
                    def stream_response_chunks():
                        full_response_accumulator = ""
                        for chunk in rag_chain.stream(response_payload):
                            if "answer" in chunk:
                                token = chunk["answer"]
                                full_response_accumulator += token
                                yield token  # Yield the token for st.write_stream
                            # Langchain's create_retrieval_chain might also yield 'context' chunks.
                            # We are primarily interested in the 'answer' for display.
                        # After streaming is complete, save the full response to history
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": full_response_accumulator}
                                ],
                            }
                        )
                        print(
                            f"DEBUG main_app: RAG chain full streamed response: {full_response_accumulator}"
                        )

                    # Use st.write_stream to display the streaming response
                    st.write_stream(stream_response_chunks)
                    # --- END STREAMING IMPLEMENTATION ---

                except Exception as e_rag:
                    st.error(f"Error during RAG chain execution: {e_rag}")
                    print(f"ERROR main_app: RAG chain execution error: {e_rag}")
                    # Add error to chat history if streaming failed partway
                    error_message_for_history = (
                        "Sorry, an error occurred while generating the response."
                    )
                    st.markdown(
                        error_message_for_history
                    )  # Display error in current chat bubble
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": error_message_for_history}
                            ],
                        }
                    )

                # No explicit st.rerun() here, st.write_stream handles its updates.
                # The full message is appended to history inside stream_response_chunks.


def main():
    """Main function to handle page routing based on login state."""
    init_db()

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    if st.session_state["logged_in"]:
        current_page_config = st.session_state.get("current_page_config")
        if current_page_config != "chatbot":
            st.set_page_config(
                page_title="DocuMentor RAG",
                page_icon="📚",
                layout="wide",
                initial_sidebar_state="expanded",
            )
            st.session_state["current_page_config"] = "chatbot"
        chatbot_page()
    elif st.session_state["page"] == "login":
        current_page_config = st.session_state.get("current_page_config")
        if current_page_config != "login":
            st.set_page_config(
                page_title="Login - DocuMentor RAG",
                layout="centered",
                initial_sidebar_state="auto",
            )
            st.session_state["current_page_config"] = "login"
        login_page()
        if st.button("Go to Register", key="goto_register_btn_login_streaming"):
            st.session_state["page"] = "register"
            st.session_state["current_page_config"] = None
            st.rerun()
    elif st.session_state["page"] == "register":
        current_page_config = st.session_state.get("current_page_config")
        if current_page_config != "register":
            st.set_page_config(
                page_title="Register - DocuMentor RAG",
                layout="centered",
                initial_sidebar_state="auto",
            )
            st.session_state["current_page_config"] = "register"
        register_page()
        if st.button("Go to Login", key="goto_login_btn_register_streaming"):
            st.session_state["page"] = "login"
            st.session_state["current_page_config"] = None
            st.rerun()
    else:
        st.session_state["page"] = "login"
        st.session_state["current_page_config"] = None
        st.rerun()


if __name__ == "__main__":
    main()
