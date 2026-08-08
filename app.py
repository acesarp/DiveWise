import streamlit as st
from pathlib import Path
from collections.abc import Callable
from services.chat_service import ChatService
from services.knowledge_base_service import KnowledgeBaseService
from models import MessageRole

st.set_page_config(page_title="DiveWise AI", page_icon="🌊", layout="wide")

@st.cache_resource
def get_knowledge_base(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    similarity_metric: str,
) -> KnowledgeBaseService:
    return KnowledgeBaseService(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        similarity_metric=similarity_metric,
    )


@st.cache_resource
def get_chat_service(
    embedding_model: str,
    generation_model: str,
    chunk_size: int,
    chunk_overlap: int,
    similarity_metric: str,
) -> ChatService:
    knowledge_base = get_knowledge_base(embedding_model, chunk_size, chunk_overlap, similarity_metric)
    return ChatService(knowledge_base=knowledge_base, generation_model=generation_model)


def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_html_fragment(file_name: str) -> None:
    fragment = Path("ui") / file_name
    st.markdown(fragment.read_text(encoding="utf-8"), unsafe_allow_html=True)


def apply_theme() -> None:
    render_html_fragment("styles.html")


def get_token_totals() -> tuple[int, int, int]:
    usage_messages = [
        message
        for message in st.session_state.messages
        if message.role == MessageRole.ASSISTANT and message.token_usage
    ]
    prompt_total = sum(message.token_usage.prompt_tokens for message in usage_messages)
    completion_total = sum(message.token_usage.completion_tokens for message in usage_messages)
    return prompt_total, completion_total, prompt_total + completion_total


def handle_document_upload() -> None:
    documents_dir = Path("data/documents")
    documents_dir.mkdir(parents=True, exist_ok=True)

    st.sidebar.subheader("Upload YOUR PDF")
    uploaded_file = st.sidebar.file_uploader(
        "Add a PDF to the knowledge base",
        type=["pdf"],
        key="sidebar_pdf_upload",
    )

    if uploaded_file is None:
        st.sidebar.caption("Upload a PDF and it will be indexed for chat answers.")
        st.sidebar.divider()
        return

    file_name = Path(uploaded_file.name).name
    destination = documents_dir / file_name

    if destination.exists():
        st.sidebar.warning("A PDF with this name already exists. Rename the file to upload it.")
        st.sidebar.divider()
        return

    if st.sidebar.button("Save PDF", key="save_uploaded_pdf", width="stretch"):
        destination.write_bytes(uploaded_file.getbuffer())
        st.sidebar.success(f"Saved {file_name}. Rebuilding knowledge base...")
        st.rerun()

    st.sidebar.caption(f"Ready to save: {file_name}")
    st.sidebar.divider()


def render_model_settings() -> tuple[str, str, int, int, str, int]:
    st.sidebar.header("⚙️ Controls")
    st.sidebar.caption("Tune model choices for retrieval and response quality.")
    embedding_model = st.sidebar.selectbox(
        "Embedding model",
        ["text-embedding-3-small", "text-embedding-3-large"],
    )
    generation_model = st.sidebar.selectbox(
        "Generation model",
        ["gpt-5-mini", "gpt-4", "gpt-4-32k"],
        index=0,
    )
    chunk_size = st.sidebar.slider(
        "Chunk size",
        min_value=300,
        max_value=1600,
        value=800,
        step=50,
        help="Controls how much text goes into each chunk before indexing.",
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk overlap",
        min_value=0,
        max_value=chunk_size - 50,
        value=min(150, chunk_size - 50),
        step=25,
        help="Controls how much text adjacent chunks share.",
    )
    similarity_metric = st.sidebar.selectbox(
        "Similarity metric",
        ["cosine", "l2", "ip"],
        index=0,
        help="Controls how vector similarity is calculated during retrieval.",
    )
    top_k = st.sidebar.slider(
        "Top-K retrieval",
        min_value=1,
        max_value=12,
        value=5,
        step=1,
        help="Controls how many chunks are retrieved to ground each answer.",
    )
    st.sidebar.divider()
    return embedding_model, generation_model, chunk_size, chunk_overlap, similarity_metric, top_k

#--------------------------------
#SIDE BAR RENDERING
def render_sidebar(knowledge_base: KnowledgeBaseService):
    """Render the sidebar with knowledge base statistics and token usage."""
    stats = knowledge_base.get_statistics()
    prompt_total, completion_total, total_tokens = get_token_totals()

    st.sidebar.subheader("Knowledge Base")
    st.sidebar.metric("Documents", stats.documents)
    st.sidebar.metric("Pages", stats.pages)
    st.sidebar.metric("Chunks", stats.chunks)
    st.sidebar.divider()

    st.sidebar.subheader("Token Usage")
    st.sidebar.metric("Prompt tokens", prompt_total)
    st.sidebar.metric("Completion tokens", completion_total)
    st.sidebar.metric("Total tokens", total_tokens)
    st.sidebar.divider()

    st.sidebar.subheader("Documents")

    for document in knowledge_base.get_documents():
        st.sidebar.markdown(f"• {document.file_name}")


def render_chat(chat_service: ChatService, top_k: int):
    st.subheader("Ask a diving question")
    st.caption("Responses are grounded in your indexed documents.")

    for message in st.session_state.messages:
        with st.chat_message(message.role.value):
            st.markdown(message.content)

            if message.sources:
                with st.expander("Sources"):

                    for source in message.sources:
                        st.markdown(f"**{source.chunk.document_name}**")
                        st.write(f"Page {source.chunk.page_number}")
                        st.caption(f"Similarity: {source.similarity:.2f}")

            if message.token_usage:
                st.caption(
                    f"Tokens - prompt: {message.token_usage.prompt_tokens}, "
                    f"completion: {message.token_usage.completion_tokens}, "
                    f"total: {message.token_usage.total_tokens}"
                )

    question = st.chat_input("Ask a scuba diving question...")

    if not question:
        return

    user_message = chat_service.create_user_message(question)

    st.session_state.messages.append(user_message    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_message = chat_service.ask(question, top_k=top_k)

        st.markdown(assistant_message.content)

        if assistant_message.sources:
            with st.expander("Sources"):
                for source in assistant_message.sources:
                    st.markdown(f"**{source.chunk.document_name}**")
                    st.write(f"Page {source.chunk.page_number}")
                    st.caption(f"Similarity: {source.similarity:.2f}")

        if assistant_message.token_usage:
            st.caption(
                f"Tokens - prompt: {assistant_message.token_usage.prompt_tokens}, "
                f"completion: {assistant_message.token_usage.completion_tokens}, "
                f"total: {assistant_message.token_usage.total_tokens}"
            )

    st.session_state.messages.append(assistant_message)


def render_documents( knowledge_base: KnowledgeBaseService):
    st.subheader("Knowledge base explorer")
    st.caption("Review indexed documents and quick previews.")

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

            st.markdown(
                f"<div class='doc-meta'>Preview from {document.file_name}</div>",
                unsafe_allow_html=True,
            )
            st.text(preview)


def render_hero() -> None:
    render_html_fragment("hero.html")


def render_overview_cards(knowledge_base: KnowledgeBaseService) -> None:
    stats = knowledge_base.get_statistics()
    prompt_total, completion_total, total_tokens = get_token_totals()

    st.markdown("<div class='stats-shell'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", stats.documents)
    col2.metric("Pages", stats.pages)
    col3.metric("Chunks", stats.chunks)
    col4.metric("Session tokens", total_tokens)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Token breakdown", expanded=False):
        c1, c2 = st.columns(2)
        c1.metric("Prompt tokens", prompt_total)
        c2.metric("Completion tokens", completion_total)


def calculate_loading_percent(current: int, total: int, message: str) -> int:
    safe_total = max(total, 1)
    ratio = current / safe_total

    if message.startswith("Reading") or message.startswith("Read"):
        return min(20 + int(ratio * 35), 55)

    if message == "Creating embeddings":
        return min(55 + int(ratio * 40), 95)

    if message == "Knowledge base ready":
        return 100

    return min(20 + int(ratio * 70), 95)


def main():
    initialize_session()
    apply_theme()

    render_hero()

    handle_document_upload()
    embedding_model, generation_model, chunk_size, chunk_overlap, similarity_metric, top_k = render_model_settings()

    progress_bar = st.progress(0, text="Checking the knowledge base...")
    percentage_text = st.empty()
    loading_status = st.status("Checking loaded documents...", expanded=True)

    try:
        document_names = tuple(sorted(path.name for path in Path("data/documents").glob("*.pdf")))
        progress_bar.progress(20, text="Loading services...")
        percentage_text.caption("20% complete")
        loading_status.update(label="Loading and indexing documents...", state="running", expanded=True)

        def update_loading_progress(current: int, total: int, message: str) -> None:
            percent = calculate_loading_percent(current, total, message)
            progress_bar.progress(percent, text=message)
            percentage_text.caption(f"{percent}% complete")
            loading_status.update(label=message, state="running", expanded=True)

        del document_names
        knowledge_base = get_knowledge_base(
            embedding_model,
            chunk_size,
            chunk_overlap,
            similarity_metric,
        )
        knowledge_base.initialize(progress_callback=update_loading_progress)
        chat_service = get_chat_service(
            embedding_model,
            generation_model,
            chunk_size,
            chunk_overlap,
            similarity_metric,
        )
        progress_bar.progress(100, text="Finalizing...")
        percentage_text.caption("100% complete")
    except RuntimeError as error:
        progress_bar.empty()
        percentage_text.empty()
        loading_status.update(label="Knowledge base could not be loaded", state="error", expanded=True)
        st.error(str(error))
        st.stop()
    progress_bar.empty()
    percentage_text.empty()
    for document in knowledge_base.get_documents():
        loading_status.write(f"✅ Loaded: {document.file_name}")
    loading_status.update(label="Knowledge base ready", state="complete", expanded=True)

    #render_sidebar(knowledge_base)
    render_overview_cards(knowledge_base)

    chat_tab, docs_tab = st.tabs(["💬 Chat", "📚 Knowledge Base"])

    with chat_tab:
        render_chat(chat_service, top_k=top_k)

    with docs_tab:
        render_documents(knowledge_base)


if __name__ == "__main__":
    main()
