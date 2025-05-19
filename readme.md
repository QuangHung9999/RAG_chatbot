# The Notgpt

Streamlit app to interact with OpenAI GPT-4o, with text, images and audio (using Whisper and TTS).

![1747619014961](image/readme/1747619014961.png)

---

## Require: python 3.12

`PATH-TO-PYTHON-312 -m venv <name_of_your_environment>`

`.venv\Scripts\Activate.ps1`

## To run the app locally:

`pip install -r requirements.txt`

or

`pip install -r requirements.txt --no-cache-dir`

`streamlit run app.py`

```
project_root/  # This is "python_ai_chat" directory
├── app/  
│   ├── __init__.py
│   ├── main_app.py      # main Streamlit UI (refactor your current main file here)
│   ├── auth.py          # SQLite auth functions
│   ├── rag_components.py # For LangChain RAG logic
│   └── utils.py         # Utilities
├── documents/           # Store PDFs here
│   └── source_document.pdf
├── vector_store_data/   #For FAISS index etc.
├── requirements.txt
└── .env
```
