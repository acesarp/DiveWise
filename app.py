import streamlit as st
from pathlib import Path
from services.chat_service import ChatService
from services.knowledge_base_service import KnowledgeBaseService

st.set_page_config(page_title="DiveWise AI", page_icon="🌊", layout="wide")

@st.cache_resource
def load_services(
    document_names: tuple[str, ...],
    embedding_model: str,
    generation_model: str,
) -> tuple[KnowledgeBaseService, ChatService]:
    # document_names and selected models are part of the cache key so changing
    # documents or model settings creates a fresh service resource.
    del document_names
    knowledge_base = KnowledgeBaseService(embedding_model=embedding_model)
    knowledge_base.initialize()

    chat_service = ChatService(knowledge_base=knowledge_base, generation_model=generation_model)

    return knowledge_base, chat_service


def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_model_settings() -> tuple[str, str]:
    st.header('⚙️ Controls')
    st.sidebar.subheader("Models")
    embedding_model = st.sidebar.selectbox("Embedding model", ["text-embedding-3-small", "text-embedding-3-large"]  )
    generation_model = st.sidebar.selectbox("Generation model", ["gpt-5-mini", "gpt-4", "gpt-4-32k"], index=0)
    st.sidebar.divider()
    return embedding_model, generation_model


def render_sidebar(knowledge_base: KnowledgeBaseService):
    stats = knowledge_base.get_statistics()

    st.sidebar.title("🌊 DiveWise AI")
    st.sidebar.multiselect("Your AI-powered scuba diving knowledge assistant.", options=[1,2,3], disabled=False)
    st.sidebar.markdown("AI-powered scuba diving assistant using Retrieval-Augmented Generation (RAG).")
    st.sidebar.divider()
    st.sidebar.subheader("Knowledge Base")
    st.sidebar.metric("Documents", stats.documents)
    st.sidebar.metric("Pages", stats.pages)
    st.sidebar.metric("Chunks", stats.chunks  )
    st.sidebar.divider()
    st.sidebar.subheader("Documents")

    for document in knowledge_base.get_documents():
        st.sidebar.write(f"📄 {document.file_name}")


def render_chat(chat_service: ChatService):
    st.header("💬 Chat")

    for message in st.session_state.messages:
        with st.chat_message(message.role.value):
            st.markdown(message.content)

            if message.sources:
                with st.expander("Sources"):

                    for source in message.sources:
                        st.markdown(f"**{source.chunk.document_name}**")
                        st.write(f"Page {source.chunk.page_number}")
                        st.caption(f"Similarity: {source.similarity:.2f}")

    question = st.chat_input("Ask a scuba diving question...")

    if not question:
        return

    user_message = chat_service.create_user_message(question)

    st.session_state.messages.append(user_message    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_message = chat_service.ask(question)

        st.markdown(assistant_message.content)

        if assistant_message.sources:
            with st.expander("Sources"):
                for source in assistant_message.sources:
                    st.markdown(f"**{source.chunk.document_name}**")
                    st.write(f"Page {source.chunk.page_number}")
                    st.caption(f"Similarity: {source.similarity:.2f}")

    st.session_state.messages.append(assistant_message)


def render_documents( knowledge_base: KnowledgeBaseService):

    st.header("📚 Knowledge Base")

    for document in knowledge_base.get_documents():
        with st.expander(document.file_name):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pages", document.page_count)

            with col2:
                st.metric("Characters", len(document.text))

            preview = document.text[:1000]

            if len(document.text) > 1000:
                preview += "..."

            st.text(preview)


def main():
    initialize_session()

    st.title("🌊 DiveWise AI")
    st.caption("Your AI-powered scuba diving knowledge assistant.")

    embedding_model, generation_model = render_model_settings()

    progress_bar = st.progress(0, text="Checking the knowledge base...")
    loading_status = st.status("Checking loaded documents...", expanded=True)

    try:
        document_names = tuple(sorted(path.name for path in Path("data/documents").glob("*.pdf")))
        progress_bar.progress(30, text="Loading services...")
        loading_status.update(label="Loading and indexing documents...", state="running", expanded=True)
        knowledge_base, chat_service = load_services(document_names, embedding_model, generation_model)
        progress_bar.progress(100, text="Finalizing...")
    except RuntimeError as error:
        progress_bar.empty()
        loading_status.update(label="Knowledge base could not be loaded", state="error", expanded=True)
        st.error(str(error))
        st.stop()
    progress_bar.empty()
    for document in knowledge_base.get_documents():
        loading_status.write(f"✅ Loaded: {document.file_name}")
    loading_status.update(label="Knowledge base ready", state="complete", expanded=True)

    render_sidebar(knowledge_base)

    chat_tab, docs_tab = st.tabs(["💬 Chat", "📚 Knowledge Base"])

    with chat_tab:
        render_chat(chat_service)

    with docs_tab:
        render_documents(knowledge_base)


if __name__ == "__main__":
    main()
