import os
from openai import OpenAI
import streamlit as st
from services.knowledge_base_service import KnowledgeBaseService
from models.chat_message import ChatMessage
from models.message_role import MessageRole

os.makedirs("uploads", exist_ok=True)

client = OpenAI( api_key=os.getenv("OPENAI_API_KEY"))

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
def load_knowledge_base():
    kb = KnowledgeBaseService()
    kb.initialize()
    return kb

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
def render_sidebar(documents):
    st.sidebar.title("🌊 DiveWise AI")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Knowledge Base")
    st.sidebar.metric(label="Documents", value=len(documents) )

    total_pages = sum(document.page_count for document in documents)

    st.sidebar.metric( label="Pages", value=total_pages)
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
    user_message = ChatMessage(
        role=MessageRole.USER,
        content=question
    )

    st.session_state.messages.append(user_message)

    for message in st.session_state.messages:
        with st.chat_message(message.role.value):
            st.markdown(message.content)

    question = st.chat_input("Ask a scuba diving question...")

    if not question:
        return

    #
    # USER MESSAGE
    #

    st.session_state.messages.append(ChatMessage(...))

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
        kb = load_knowledge_base()

    initialize_chat()
    render_sidebar(kb.get_documents())

    tab1, tab2 = st.tabs(
        [
            "💬 Chat",
            "📚 Knowledge Base"
        ]
    )

    with tab1:
        render_chat()

    with tab2:
        render_documents(kb.get_documents())

if __name__ == "__main__":
    main()