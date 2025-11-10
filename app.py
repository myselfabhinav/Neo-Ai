import os
import sys
import streamlit as st
import logging
from dotenv import load_dotenv
import tempfile
import speech_recognition as sr

# --------------------------- Setup --------------------------- #
print("🧠 Streamlit using Python:", sys.executable)

env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path, override=True)
else:
    print("⚠️ .env file not found!")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# --------------------------- Safe Imports --------------------------- #
try:
    from models.llm import (
        get_openai_model,
        get_gemini_model,
        invoke_model,
        get_free_embeddings
    )
    from utils.web_search import get_web_results_text
    from utils.helpers import format_response
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
except Exception:
    def get_openai_model(): return "openai-fallback"
    def get_gemini_model(): return "gemini-fallback"
    def invoke_model(**kwargs): return "Mock AI reply."
    def get_free_embeddings(): raise RuntimeError("Embeddings unavailable")

# --------------------------- Logging --------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

# --------------------------- Chat Response --------------------------- #
def get_chat_response(provider, chat_model, messages, system_prompt, response_mode, enable_rag, enable_web_only=False):
    try:
        user_query = messages[-1]["content"]
        context_text = ""

        if enable_rag and "vectorstore" in st.session_state:
            try:
                docs = list(st.session_state["vectorstore"].docstore._dict.values())
                context_text = "\n\n".join([d.page_content for d in docs])
            except Exception as e:
                logging.warning(f"Failed to retrieve RAG context: {e}")

        if enable_web_only:
            web_data = get_web_results_text(user_query)
            return web_data if web_data.strip() else "No relevant web results found."

        full_prompt = "You are a helpful assistant providing accurate answers."
        if context_text:
            full_prompt += f"\n\nContext:\n{context_text}"
        if response_mode == "Concise":
            full_prompt += "\nRespond concisely under 100 words."
        else:
            full_prompt += "\nProvide a detailed and structured response."

        response = invoke_model(
            model=chat_model, messages=messages,
            provider=provider, system_prompt=full_prompt
        )
        return format_response(str(response).strip(), mode=response_mode.lower())

    except Exception as e:
        return f"⚠️ Error while generating response: {e}"

# --------------------------- Pages --------------------------- #
def instructions_page():
    st.markdown("""
    <div class="fade-in">
    <h1>📘 AI ChatBot Setup Guide</h1>
    <p>Welcome to your <b>NeoStats AI Assistant</b>! Here's what you can do:</p>
    <ul>
        <li>🧠 Talk with Gemini or OpenAI models</li>
        <li>🎙️ Use live voice input</li>
        <li>📚 Upload PDFs for smart context (RAG)</li>
        <li>🌐 Enable Web Search mode</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# --------------------------- Chat Page --------------------------- #
def chat_page():
    st.markdown("""
    <div class="header-glow">
        <h1>🤖 NeoStats AI ChatBot</h1>
        <p>Ask anything — powered by advanced AI engines</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("<h2>⚙️ Settings</h2>", unsafe_allow_html=True)
        provider = st.selectbox("LLM Provider", ["Gemini", "OpenAI"])
        response_mode = st.radio("Response Mode", ["Concise", "Detailed"])
        enable_rag = st.checkbox("Enable RAG (PDF Context)")
        enable_web_only = st.checkbox("Enable Web Search Only")

        try:
            chat_model = {
                "Gemini": get_gemini_model,
                "OpenAI": get_openai_model
            }[provider]()
            st.markdown(
                f"""
                <div class="model-status">
                    <div class="pulse"></div>
                    <span class="ready-text"> {provider} Model Ready</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Model load failed: {e}")
            chat_model = "fallback"

    # PDF Upload
    with st.expander("📄 Upload PDF for Context"):
        uploaded_pdf = st.file_uploader("Choose a PDF", type=["pdf"])
        if uploaded_pdf:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name
            try:
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = splitter.split_documents(pages)
                embeddings = get_free_embeddings()
                st.session_state["vectorstore"] = FAISS.from_documents(docs, embeddings)
                st.info("✅ PDF processed successfully!")
            except Exception as e:
                st.error(f"PDF processing failed: {e}")

    # Chat memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_placeholder = st.container()
    with chat_placeholder:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ---------------- Fixed Input Bar ----------------
    st.markdown("""
    <style>
    /* ---- Background animation ---- */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1e3c72);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        z-index: -1;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .model-status {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(0,255,255,0.1);
        padding: 10px;
        border-radius: 10px;
        margin-top: 10px;
        box-shadow: 0 0 15px rgba(0,255,255,0.3);
    }
    .pulse {
        width: 15px;
        height: 15px;
        border-radius: 50%;
        background: #00e0ff;
        box-shadow: 0 0 15px #00e0ff;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.9); opacity: 0.7; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(0.9); opacity: 0.7; }
    }
    .ready-text {
        color: #00e0ff;
        font-weight: bold;
    }

    .header-glow h1 {
        text-align: center;
        color: #fff;
        text-shadow: 0 0 20px #00e0ff;
    }
    .header-glow p {
        text-align: center;
        color: #bbb;
    }

    div[data-testid="stBottomBlockContainer"] {
        position: fixed !important;
        bottom: 0;
        width: 100%;
        background: rgba(0, 0, 0, 0.85);
        backdrop-filter: blur(12px);
        padding: 0.7rem 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

    # Chat input row
    col1, col2 = st.columns([10, 1])
    with col1:
        prompt = st.chat_input("💬 Type your message here...")
    with col2:
        mic_pressed = st.button("🎤", key="mic_btn", help="Speak now", use_container_width=True)

    # Voice Input
    if mic_pressed:
        recognizer = sr.Recognizer()
        with st.spinner("🎧 Listening..."):
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    voice_text = recognizer.recognize_google(audio)
                    st.session_state.messages.append({"role": "user", "content": voice_text})
                    st.success(f"🗣️ You said: {voice_text}")
            except Exception as e:
                st.error(f"Voice error: {e}")

    # Handle text input
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate AI response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with chat_placeholder.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = get_chat_response(
                    provider, chat_model, st.session_state.messages,
                    "", response_mode, enable_rag, enable_web_only
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        if "vectorstore" in st.session_state:
            del st.session_state["vectorstore"]
        st.rerun()

# --------------------------- Main --------------------------- #
def main():
    st.set_page_config(page_title="AI ChatBot ✨", layout="wide", page_icon="🤖")
    with st.sidebar:
        st.markdown("<h1>📍 Navigation</h1>", unsafe_allow_html=True)
        page = st.radio("Go to:", ["Chat", "Instructions"])

    if page == "Chat":
        chat_page()
    else:
        instructions_page()

if __name__ == "__main__":
    main()
