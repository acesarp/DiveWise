import os
from openai import OpenAI
from dotenv import load_dotenv

class OpenAIService:

    def __init__(self, model: str = "gpt-5-mini"):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        print(f"OPENAI_API_KEY: {api_key}")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured. Add it to .env or the environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = model


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
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text
