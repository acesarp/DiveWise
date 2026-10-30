# DiveWise AI

A retrieval-augmented generation (RAG) assistant for scuba diving. DiveWise reads PDF documents (manuals, guides, dive tables), turns them into a searchable knowledge base and answers questions using only what's in those documents, so the answers stay grounded instead of being made up by the model.

The pipeline is written from scratch in Python (document loading, cleaning, chunking, embeddings, vector search and generation), without a RAG framework such as LangChain, to show how each step works.

## Features

- Ask questions in a chat interface and get answers based only on the indexed documents
- The assistant says it doesn't know when the answer isn't in the context
- Upload your own PDF from the sidebar; it's indexed and added to the knowledge base
- Tune the pipeline live from the sidebar:
  - embedding model (`text-embedding-3-small` or `text-embedding-3-large`)
  - generation model
  - chunk size and chunk overlap
  - similarity metric (cosine, L2 or inner product)
  - number of chunks retrieved (top-k)
- Each combination of settings gets its own vector collection, so you can compare results without rebuilding everything
- Knowledge base statistics (documents, pages) and token usage for each conversation
- Optional password gate, since every question is billed to the OpenAI account
- Containerized with Docker and deployed automatically on every push to `main`

## How it works

```
PDFs -> DocumentLoader -> TextCleaner -> Chunker -> EmbeddingService -> VectorStore (ChromaDB)
                                                                              |
Question -> EmbeddingService -> RetrievalService (top-k chunks) --------------'
                                       |
                                 OpenAIService (answer grounded in the retrieved chunks)
```

1. **Load:** `DocumentLoader` reads the PDFs in `data/documents`.
2. **Clean:** `TextCleaner` normalizes the extracted text.
3. **Chunk:** `Chunker` splits the text into overlapping chunks (800 characters with 150 of overlap by default).
4. **Embed:** `EmbeddingService` creates an embedding for each chunk with the OpenAI API.
5. **Store:** `VectorStore` saves the chunks and embeddings in a persistent ChromaDB collection.
6. **Retrieve:** for each question, `RetrievalService` embeds the question and finds the most similar chunks.
7. **Generate:** `OpenAIService` sends the retrieved chunks and the question to the model, with instructions to answer only from that context.

## Tech stack

| Area | Technologies |
|---|---|
| Language | Python 3.12 |
| UI | Streamlit |
| LLM and embeddings | OpenAI API (Responses API, text-embedding-3) |
| Vector database | ChromaDB (persistent client) |
| Documents | pypdf |
| Token counting | tiktoken |
| DevOps | Docker, Docker Compose, GitHub Actions, self-hosted runner |
| Code quality | pylint, flake8 |

## Project structure

```
app.py                  Streamlit UI: chat, PDF upload, pipeline controls
models.py               Data models (Document, Chunk, SearchResult, ChatMessage, TokenUsage)
services/
  document_loader.py    Reads PDFs
  text_cleaner.py       Cleans extracted text
  chunker.py            Splits text into overlapping chunks
  embedding_service.py  OpenAI embeddings
  vector_store.py       ChromaDB storage and search
  retrieval_service.py  Finds the most relevant chunks for a question
  openai_service.py     Builds the prompt and generates the answer
  chat_service.py       Ties retrieval and generation together
  knowledge_base_service.py  Builds and manages the knowledge base
dockerfile
compose.yml
.github/workflows/      CI/CD pipeline
```

## Getting started

### Prerequisites

- Python 3.12, or Docker
- An OpenAI API key

### Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=<your-openai-api-key>
APP_PASSWORD=<optional-password>
```

If `APP_PASSWORD` is not set, the app is open to anyone who can reach it.

Put your PDFs in `data/documents/`, or upload them from the sidebar once the app is running.

### Run with Docker Compose

```bash
docker compose up -d --build
```

Open http://localhost:3001

The uploaded documents and the vector store are kept in Docker volumes, so they survive restarts and redeploys.

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## CI/CD

The GitHub Actions workflow (`.github/workflows/streamlit-app.yml`) runs on every push and pull request to `main`:

1. **Checks:** installs the dependencies and runs pylint and flake8.
2. **Deploy** (pushes to `main` only): a self-hosted runner on the home server writes the `.env` file from GitHub secrets (readable only by the runner user), builds the image, restarts the container with Docker Compose, waits for Streamlit's health check to respond and then deletes the `.env` file.

The Docker image also has a `HEALTHCHECK` that calls Streamlit's health endpoint.
