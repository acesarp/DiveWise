import os
from openai import OpenAI

class OpenAIService:

    def __init__(self):
        api_key=os.getenv("OPENAI_API_KEY")
        print(api_key)
        self.client = OpenAI(api_key=api_key)


    def generate(self, question: str, search_results: list) -> str:
        context = "\n\n".join(result.chunk.text  for result in search_results)

        prompt = f"""
You are an expert scuba diving assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
        response = self.client.responses.create(model="gpt-5", input=prompt,)
        return response.output_text