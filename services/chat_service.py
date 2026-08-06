from models.chat_message import ChatMessage
from models.message_role import MessageRole
from services.retrieval_service import RetrievalService
from services.openai_service import OpenAIService


class ChatService:

    def __init__(self):

        self.retrieval_service = RetrievalService()
        self.openai_service = OpenAIService()

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

        return ChatMessage(         role=MessageRole.ASSISTANT,            content=answer,            sources=search_results        )