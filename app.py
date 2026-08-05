import base64
from io import BytesIO
import os
import time
from openai import OpenAI
import streamlit as st
from services.document_loader import DocumentLoader
from services.vector_store import VectorStore
from services.chat_service import ChatService
from services.chat_message import ChatMessage

os.makedirs("uploads", exist_ok=True)

loader = DocumentLoader()

documents = loader.load_documents()

for document in documents:
    print(document.file_name)
    print(document.page_count)
    print(len(document.text))


client = OpenAI( api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="DiveWise AI",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 DiveWise AI")
st.caption("Your AI-powered scuba diving knowledge assistant")

with st.spinner("Loading knowledge base..."):
    documents = loader.load_documents()

st.success(f"Loaded {len(documents)} documents.")

st.divider()

for document in documents:
    with st.expander(document.file_name):
        st.write(f"**Pages:** {document.page_count}")
        st.write(f"**Characters:** {len(document.text):,}")

        st.text(document.text[:500] + "...")


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="DiveWise AI",
    page_icon="🌊",
    layout="wide"
)


# ---------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------

@st.cache_resource
def load_documents():
    loader = DocumentLoader()
    return loader.load_documents()


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

def render_sidebar(documents):

    st.sidebar.title("🌊 DiveWise AI")

    st.sidebar.markdown("---")

    st.sidebar.subheader("Knowledge Base")

    st.sidebar.metric(
        label="Documents",
        value=len(documents)
    )

    total_pages = sum(document.page_count for document in documents)

    st.sidebar.metric(
        label="Pages",
        value=total_pages
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Loaded Documents")

    for document in documents:
        st.sidebar.write(f"📄 {document.file_name}")


def render_documents(documents):

    st.header("Knowledge Base")

    for document in documents:
        with st.expander(document.file_name):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Pages:** {document.page_count}")

            with col2:
                st.write(f"**Characters:** {len(document.text):,}")

            preview = document.text[:600]

            if len(document.text) > 600:
                preview += "..."

            st.text(preview)


def initialize_chat():

    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_chat():

    st.header("Chat")

    for message in st.session_state.messages:

        with st.chat_message(message.role.value):
            st.markdown(message.content)

    question = st.chat_input(
        "Ask a scuba diving question..."
    )

    if not question:
        return

    #
    # USER MESSAGE
    #

    st.session_state.messages.append(
        ChatMessage(...)
    )

    #
    # TODO
    #
    retrieval_service.retrieve(question)
    
     openai_service.generate(...)
    
     st.session_state.messages.append(response)
    


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    st.title("🌊 DiveWise AI")

    st.caption(
        "AI-powered scuba diving assistant built with RAG, OpenAI and ChromaDB."
    )

    with st.spinner("Loading knowledge base..."):

        documents = load_documents()

    initialize_chat()

    render_sidebar(documents)

    tab1, tab2 = st.tabs(
        [
            "💬 Chat",
            "📚 Knowledge Base"
        ]
    )

    with tab1:

        render_chat()

    with tab2:

        render_documents(documents)


if __name__ == "__main__":
    main()