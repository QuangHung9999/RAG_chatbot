import streamlit as st
import os
import dotenv
from datetime import datetime

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
    get_all_document_names,
    resync_knowledge_base,
    remove_document_from_kb,
    get_knowledge_base_stats,
)
from app.config import (
    GOOGLE_API_KEY,
    GOOGLE_MODELS,
    APP_TITLE,
    APP_SUBTITLE,
    APP_ICON,
    DEFAULT_TEMPERATURE,
    RETRIEVER_K
)
# Import conversation tracking functionality
from app.conversation_tracker import ConversationTracker, Timer, estimate_tokens

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Using models from config


def chatbot_page():
    """Main page for the RAG chatbot interface."""
    st.markdown(
        f"<h1 style='text-align: center; color: orange;'>{APP_TITLE}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align: center;'>{APP_SUBTITLE}</p>",
        unsafe_allow_html=True,
    )

    # Initialize session state flags if they don't exist
    default_keys = {
        "google_api_key": os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY),
        "messages": [],
        "embeddings_model_loaded": False,
        "embeddings_model": None,
        "embedding_model_load_attempted": False,
        "vector_store_initialized": False,
        "vector_store": None,
        "vector_store_init_attempted": False,
        # Add conversation tracking session state
        "conversation_tracker": None,
        "current_conversation_id": None,
    }
    for key, value in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize conversation tracker if not already done
    if st.session_state.conversation_tracker is None:
        st.session_state.conversation_tracker = ConversationTracker()

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
            # End current conversation if one exists
            if st.session_state.current_conversation_id is not None:
                if st.session_state.conversation_tracker:
                    st.session_state.conversation_tracker.end_conversation(
                        st.session_state.current_conversation_id
                    )
                st.session_state.current_conversation_id = None
            
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
            GOOGLE_MODELS,
            index=0,
            key="llm_model_select_ui_streaming",
        )
        temperature = st.slider(
            "Temperature (Creativity):",
            0.0,
            1.0,
            DEFAULT_TEMPERATURE,
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
                
                # Add KB Stats display at the top of the sidebar
                with st.sidebar.expander("📊 Knowledge Base Stats", expanded=False):
                    # Get live stats from vector store
                    kb_stats = get_knowledge_base_stats()
                    # Update vector count if vector store is loaded
                    if st.session_state.vector_store and hasattr(st.session_state.vector_store, 'index') and st.session_state.vector_store.index:
                        kb_stats["vector_count"] = st.session_state.vector_store.index.ntotal
                    
                    st.write("**KB Statistics:**")
                    st.write(f"• Documents: {kb_stats['document_count']}")
                    st.write(f"• Vector Count: {kb_stats['vector_count']}")
                    st.write(f"• Vector Store Size: {kb_stats['vector_store_size_mb']} MB")
                    
                    # Last sync time (could be stored in session state when synced)
                    if "last_sync_time" in st.session_state:
                        st.write(f"• Last Synced: {st.session_state['last_sync_time']}")
                    else:
                        st.write("• Last Synced: Unknown")
                
                # Add conversation metrics dashboard
                with st.sidebar.expander("📈 Conversation Metrics", expanded=False):
                    if st.session_state.current_conversation_id:
                        # Get metrics for current conversation
                        conversation_stats = st.session_state.conversation_tracker.get_conversation_stats(
                            conversation_id=st.session_state.current_conversation_id
                        )
                        
                        if conversation_stats:
                            conv_data = conversation_stats[0]
                            st.write("**Current Conversation:**")
                            st.write(f"• User: {conv_data['user_id']}")
                            st.write(f"• Messages: {conv_data['total_messages']}")
                            st.write(f"• Total Tokens: {conv_data['total_tokens']}")
                            
                            avg_latency = 0
                            if conv_data['total_messages'] > 0:
                                avg_latency = conv_data['total_latency_ms'] / conv_data['total_messages']
                            st.write(f"• Avg Latency: {avg_latency:.0f} ms")
                            
                            # Get conversation details to display more metrics
                            conv_details = st.session_state.conversation_tracker.get_conversation_details(
                                st.session_state.current_conversation_id
                            )
                            
                            if conv_details and 'messages' in conv_details:
                                # Calculate average metrics
                                assist_msgs = [m for m in conv_details['messages'] if m['message_type'] == 'assistant']
                                if assist_msgs:
                                    avg_gen_time = sum(m['generation_time_ms'] for m in assist_msgs) / len(assist_msgs)
                                    avg_ret_time = sum(m['retrieval_time_ms'] for m in assist_msgs) / len(assist_msgs)
                                    avg_docs = sum(m['documents_retrieved'] for m in assist_msgs) / len(assist_msgs)
                                    
                                    st.write("**Performance:**")
                                    st.write(f"• Avg Generation: {avg_gen_time:.0f} ms")
                                    st.write(f"• Avg Retrieval: {avg_ret_time:.0f} ms")
                                    st.write(f"• Avg Docs Retrieved: {avg_docs:.1f}")
                    else:
                        st.write("No active conversation.")
                    
                    # Overall stats
                    aggregated_stats = st.session_state.conversation_tracker.get_aggregated_stats()
                    if aggregated_stats:
                        st.write("**Overall Usage:**")
                        st.write(f"• Total Conversations: {aggregated_stats.get('conversation_count', 0)}")
                        st.write(f"• Total Messages: {aggregated_stats.get('total_messages', 0)}")
                        st.write(f"• Total Tokens: {aggregated_stats.get('total_tokens', 0)}")
                
                # Add knowledge base management section
                with st.sidebar.expander("📚 Knowledge Base Management", expanded=True):
                    # Show list of available documents
                    if "kb_documents" not in st.session_state:
                        st.session_state.kb_documents = get_all_document_names()
                    
                    # Initialize states for document processing
                    if "files_processed" not in st.session_state:
                        st.session_state.files_processed = False
                    if "document_removed" not in st.session_state:
                        st.session_state.document_removed = False
                        st.session_state.removed_document_name = ""
                    if "kb_synced" not in st.session_state:
                        st.session_state.kb_synced = False
                    
                    # Display appropriate action buttons for KB management
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🔄 Refresh List", key="refresh_docs_btn"):
                            st.session_state.kb_documents = get_all_document_names()
                            st.rerun()
                    
                    with col2:
                        # Show the sync button only if we're not in post-sync state
                        if not st.session_state.kb_synced:
                            if st.button("🔄 Full KB Sync", key="resync_kb_btn"):
                                with st.spinner("Syncing knowledge base..."):
                                    updated_vector_store = resync_knowledge_base(st.session_state.embeddings_model)
                                    if updated_vector_store:
                                        st.session_state.vector_store = updated_vector_store
                                        st.session_state.kb_documents = get_all_document_names()
                                        # Record sync time
                                        st.session_state["last_sync_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        # Set the sync state
                                        st.session_state.kb_synced = True
                                        st.rerun()
                        else:
                            # Show success message with an OK button to reset the state
                            st.success("Knowledge base synced successfully!")
                            if st.button("OK", key="sync_ok_btn"):
                                st.session_state.kb_synced = False
                                st.rerun()
                    
                    # If a document was just removed, show a success message
                    if st.session_state.document_removed:
                        st.success(f"Document '{st.session_state.removed_document_name}' was removed successfully!")
                        if st.button("OK", key="doc_removed_ok_btn"):
                            st.session_state.document_removed = False
                            st.rerun()
                    # Display document list with options if no document was just removed
                    elif st.session_state.kb_documents:
                        st.write("**Available Documents:**")
                        for doc in st.session_state.kb_documents:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"- {doc}")
                            with col2:
                                if st.button("🗑️", key=f"remove_{doc}", help=f"Remove {doc} from knowledge base"):
                                    with st.spinner(f"Removing {doc}..."):
                                        updated_vector_store = remove_document_from_kb(
                                            doc, 
                                            st.session_state.vector_store,
                                            st.session_state.embeddings_model
                                        )
                                        if updated_vector_store:
                                            st.session_state.vector_store = updated_vector_store
                                            st.session_state.kb_documents = get_all_document_names()
                                            # Record the update time
                                            st.session_state["last_sync_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            # Set document removal state
                                            st.session_state.document_removed = True
                                            st.session_state.removed_document_name = doc
                                            st.rerun()
                    else:
                        st.info("No documents in knowledge base. Upload documents below.")
                    
                    # File upload section
                    st.write("**Upload New Documents:**")
                    
                    # Only show the uploader if we're not in a state right after processing
                    if not st.session_state.files_processed:
                        uploaded_files = st.file_uploader(
                            "Select PDF files to add",
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
                                # Update document list after adding new files
                                st.session_state.kb_documents = get_all_document_names()
                                # Record the update time
                                st.session_state["last_sync_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                # Set the processed flag to true to hide the uploader on next render
                                st.session_state.files_processed = True
                                st.rerun()
                            else:
                                st.sidebar.error(
                                    "Embedding model not available for adding documents. Please retry loading."
                                )
                    else:
                        # Show a success message and a button to upload more files
                        st.success("Files uploaded successfully!")
                        if st.button("Upload More Files", key="upload_more_btn"):
                            st.session_state.files_processed = False
                            st.rerun()

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
                        # Start conversation if not already started
                        if st.session_state.current_conversation_id is None:
                            user_id = st.session_state.get("username", "Guest")
                            st.session_state.current_conversation_id = st.session_state.conversation_tracker.start_conversation(user_id)
                        
                        # Track user message with metrics
                        overall_timer = Timer().start()
                        retrieval_timer = Timer().start()
                        
                        # Get number of retrieved documents
                        num_docs_retrieved = len(retrieved_docs) if retrieved_docs else 0
                        retrieval_time = retrieval_timer.stop().elapsed_ms()
                        
                        # Add user message to conversation history
                        user_metrics = {
                            'tokens': estimate_tokens(prompt),
                            'retrieval_time_ms': retrieval_time,
                            'documents_retrieved': num_docs_retrieved
                        }
                        st.session_state.conversation_tracker.add_message(
                            st.session_state.current_conversation_id,
                            'user',
                            prompt,
                            user_metrics
                        )
                        
                        # Begin tracking assistant response
                        generation_timer = Timer().start()
                        full_response_accumulator = ""
                        
                        for chunk in rag_chain.stream(response_payload):
                            if "answer" in chunk:
                                token = chunk["answer"]
                                full_response_accumulator += token
                                yield token  # Yield the token for st.write_stream
                        
                        # After streaming is complete, record metrics
                        generation_time = generation_timer.stop().elapsed_ms()
                        total_latency = overall_timer.stop().elapsed_ms()
                        
                        # Add assistant message to conversation history with metrics
                        assistant_metrics = {
                            'tokens': estimate_tokens(full_response_accumulator),
                            'latency_ms': total_latency,
                            'generation_time_ms': generation_time,
                            'retrieval_time_ms': retrieval_time,
                            'documents_retrieved': num_docs_retrieved
                        }
                        
                        st.session_state.conversation_tracker.add_message(
                            st.session_state.current_conversation_id,
                            'assistant',
                            full_response_accumulator,
                            assistant_metrics
                        )
                        
                        # Save the full response to chat history
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
                        print(f"DEBUG main_app: Response metrics - Tokens: {assistant_metrics['tokens']}, " 
                              f"Latency: {assistant_metrics['latency_ms']}ms, "
                              f"Generation: {assistant_metrics['generation_time_ms']}ms, "
                              f"Retrieval: {assistant_metrics['retrieval_time_ms']}ms, "
                              f"Docs: {assistant_metrics['documents_retrieved']}")

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


def admin_page():
    """Admin page for viewing conversation analytics and system metrics."""
    st.title("Admin Dashboard")
    
    # Check if user has admin privileges (simple check for demo)
    if st.session_state.get("username") != "admin":
        st.warning("You need admin privileges to access this page.")
        if st.button("Back to Chat"):
            st.session_state["page"] = "chatbot"
            st.rerun()
        return
    
    # Navigation
    if st.button("Back to Chat"):
        st.session_state["page"] = "chatbot"
        st.rerun()
    
    # Initialize tracker if needed
    if st.session_state.conversation_tracker is None:
        st.session_state.conversation_tracker = ConversationTracker()
    
    # Show analytics
    st.header("Conversation Analytics")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=None)
    with col2:
        end_date = st.date_input("End Date", value=None)
    
    # Convert to datetime if selected
    start_datetime = None
    end_datetime = None
    if start_date:
        start_datetime = datetime.combine(start_date, datetime.min.time())
    if end_date:
        end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Get stats based on date range
    aggregated_stats = st.session_state.conversation_tracker.get_aggregated_stats(
        start_date=start_datetime,
        end_date=end_datetime
    )
    
    # Display overall metrics
    st.subheader("System Metrics")
    if aggregated_stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Conversations", aggregated_stats.get('conversation_count', 0))
        with col2:
            st.metric("Total Messages", aggregated_stats.get('total_messages', 0))
        with col3:
            st.metric("Total Tokens", aggregated_stats.get('total_tokens', 0))
        
        col1, col2 = st.columns(2)
        with col1:
            avg_tokens = aggregated_stats.get('avg_tokens_per_conversation', 0)
            st.metric("Avg Tokens per Conversation", f"{avg_tokens:.0f}")
        with col2:
            avg_latency = aggregated_stats.get('avg_latency_ms_per_conversation', 0)
            st.metric("Avg Latency (ms)", f"{avg_latency:.0f}")
            
        # Add visualizations for metrics over time
        try:
            # Get all conversations for time-based visualization
            all_convs = st.session_state.conversation_tracker.get_conversation_stats(
                start_date=start_datetime,
                end_date=end_datetime
            )
            
            if all_convs and len(all_convs) > 1:
                import pandas as pd
                import altair as alt
                
                # Create dataframe for time series
                time_data = []
                for conv in all_convs:
                    start_time = conv.get('start_time')
                    if isinstance(start_time, str):
                        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    
                    time_data.append({
                        'Date': start_time,
                        'Tokens': conv.get('total_tokens', 0),
                        'Messages': conv.get('total_messages', 0),
                        'Latency (ms)': conv.get('total_latency_ms', 0)
                    })
                
                if time_data:
                    df_time = pd.DataFrame(time_data)
                    df_time = df_time.sort_values('Date')
                    
                    # Token usage over time
                    st.subheader("Token Usage Over Time")
                    token_chart = alt.Chart(df_time).mark_line().encode(
                        x='Date:T',
                        y='Tokens:Q',
                        tooltip=['Date', 'Tokens']
                    ).properties(height=300)
                    st.altair_chart(token_chart, use_container_width=True)
                    
                    # Latency over time 
                    st.subheader("Response Latency Over Time")
                    latency_chart = alt.Chart(df_time).mark_line().encode(
                        x='Date:T',
                        y='Latency (ms):Q',
                        tooltip=['Date', 'Latency (ms)']
                    ).properties(height=300)
                    st.altair_chart(latency_chart, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create visualizations: {e}")
    else:
        st.info("No conversation data available for the selected period.")
    
    # List all conversations
    st.subheader("Recent Conversations")
    conversations = st.session_state.conversation_tracker.get_conversation_stats(
        start_date=start_datetime,
        end_date=end_datetime
    )
    
    if conversations:
        # Create a dataframe for display
        import pandas as pd
        
        df_data = []
        for conv in conversations:
            # Parse dates from string format if needed
            start_time = conv.get('start_time')
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            
            end_time = conv.get('end_time')
            if end_time:
                if isinstance(end_time, str):
                    end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                duration = (end_time - start_time).total_seconds()
            else:
                duration = None
            
            df_data.append({
                'ID': conv.get('id'),
                'User': conv.get('user_id'),
                'Start Time': start_time,
                'Duration (s)': duration,
                'Messages': conv.get('total_messages'),
                'Tokens': conv.get('total_tokens'),
                'Avg Latency (ms)': conv.get('total_latency_ms', 0) / max(1, conv.get('total_messages', 1))
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df)
        
        # Conversation details view
        selected_conv_id = st.selectbox(
            "Select conversation to view details",
            options=[c.get('id') for c in conversations],
            format_func=lambda x: f"Conversation {x}"
        )
        
        if selected_conv_id:
            conv_details = st.session_state.conversation_tracker.get_conversation_details(selected_conv_id)
            if conv_details and 'messages' in conv_details:
                st.subheader(f"Conversation #{selected_conv_id} Details")
                
                # Show messages in this conversation
                for msg in conv_details['messages']:
                    with st.expander(f"{msg['message_type'].capitalize()} ({msg['timestamp']})", expanded=False):
                        st.write(msg['content'])
                        st.write("---")
                        # Display metrics for this message
                        cols = st.columns(4)
                        cols[0].metric("Tokens", msg['tokens'])
                        cols[1].metric("Latency (ms)", msg['latency_ms'])
                        cols[2].metric("Generation (ms)", msg['generation_time_ms'])
                        cols[3].metric("Retrieval (ms)", msg['retrieval_time_ms'])
    else:
        st.info("No conversations found for the selected period.")


def main():
    """Main function to handle page routing based on login state."""
    init_db()

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    if st.session_state["logged_in"]:
        if st.session_state["page"] == "admin":
            current_page_config = st.session_state.get("current_page_config")
            if current_page_config != "admin":
                st.set_page_config(
                    page_title="Admin - DocuMentor RAG",
                    page_icon="📊",
                    layout="wide",
                    initial_sidebar_state="collapsed",
                )
                st.session_state["current_page_config"] = "admin"
            admin_page()
        else:  # Default to chatbot page
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
            # Add a button to go to admin page
            if st.session_state.get("username") == "admin":
                with st.sidebar:
                    if st.button("Admin Dashboard"):
                        st.session_state["page"] = "admin"
                        st.rerun()
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
