import os

from openai import OpenAI


class EmbeddingService:

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") )
        self.model = "text-embedding-3-small"


    def create_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


    def create_embeddings(self, chunks) -> list[list[float]]:
        embeddings = []

        for chunk in chunks:
            embeddings.append(self.create_embedding(chunk.text) )

        return embeddings