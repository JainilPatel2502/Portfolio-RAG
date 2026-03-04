from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from typing import List, Dict, Any, TypedDict, Annotated , Optional
import requests
import os
from dotenv import load_dotenv

load_dotenv()
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

llm = ChatOpenRouter(
    model="google/gemma-3-27b-it:free",
)
# class OpenRouterEmbeddings(Embeddings):
#     def __init__(self, model="nvidia/llama-nemotron-embed-vl-1b-v2:free"):
#         self.model = model
#         self.api_key = os.getenv("OPENROUTER_API_KEY")
    
#     def embed_documents(self, texts):
#         response = requests.post(
#             "https://openrouter.ai/api/v1/embeddings",
#             headers={"Authorization": f"Bearer {self.api_key}"},
#             json={"model": self.model, "input": texts}
#         )
#         return [r["embedding"] for r in response.json()["data"]]
    
#     def embed_query(self, text):
#         return self.embed_documents([text])[0]

# embeddings = OpenRouterEmbeddings()

# class LMStudioEmbeddings(Embeddings):
#     def __init__(self, endpoint_url="http://localhost:1234/v1/embeddings"):
#         self.endpoint_url = endpoint_url

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         response = requests.post(
#             self.endpoint_url,
#             json={"input": texts}
#         )
#         response.raise_for_status()
#         data = response.json()
#         return [r["embedding"] for r in data["data"]]

#     def embed_query(self, text: str) -> List[float]:
#         return self.embed_documents([text])[0]
# embeddings = LMStudioEmbeddings()

# llm = ChatOpenAI(
#     openai_api_base="http://localhost:1234/v1",
#     openai_api_key="lm-studio",
#     model="google/gemma-3n-e4b", 
#     temperature=0.7
# )


