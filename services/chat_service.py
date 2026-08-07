from models import ChatMessage, MessageRole
from services.retrieval_service import RetrievalService
from services.openai_service import OpenAIService


class ChatService:

    def __init__(self, knowledge_base):
        self.retrieval_service = RetrievalService(
            embedding_service=knowledge_base.embedding_service,
            vector_store=knowledge_base.vector_store,
        )
        self.openai_service = OpenAIService()

    @staticmethod
    def create_user_message(question: str) -> ChatMessage:
        return ChatMessage(role=MessageRole.USER, content=question)

    def ask(self, question: str) -> ChatMessage:
        """
        Answers a user's question using the RAG pipeline.

        Args:
            question: User's question.

        Returns:
            ChatMessage containing the assistant's response.
        """

        # Retrieve relevant chunks
        search_results = self.retrieval_service.retrieve(question)

        # Generate response
        answer = self.openai_service.generate(question=question,search_results=search_results)

        return ChatMessage(role=MessageRole.ASSISTANT, content=answer, sources=search_results)
