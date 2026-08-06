import streamlit as st

from services.chat_service import ChatService
from services.knowledge_base_service import KnowledgeBaseService


st.set_page_config(page_title="DiveWise AI", page_icon="🌊", layout="wide")


@st.cache_resource
def load_services() -> tuple[KnowledgeBaseService, ChatService]:

    knowledge_base = KnowledgeBaseService()
    knowledge_base.initialize()

    chat_service = ChatService(knowledge_base=knowledge_base)

    return knowledge_base, chat_service


def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar(knowledge_base: KnowledgeBaseService):

    stats = knowledge_base.get_statistics()

    st.sidebar.title("🌊 DiveWise AI")
    st.sidebar.markdown("AI-powered scuba diving assistant using Retrieval-Augmented Generation (RAG)."    )
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

    st.session_state.messages.append(       assistant_message   )


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

    with st.spinner( "Loading knowledge base..." ):
        knowledge_base, chat_service = (load_services())

    st.title("🌊 DiveWise AI")
    st.caption("Your AI-powered scuba diving knowledge assistant."    )

    render_sidebar(knowledge_base    )

    chat_tab, docs_tab = st.tabs(["💬 Chat", "📚 Knowledge Base"]    )

    with chat_tab:
        render_chat(chat_service)

    with docs_tab:
        render_documents(knowledge_base)


if __name__ == "__main__":
    main()