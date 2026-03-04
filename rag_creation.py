
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import os
from typing import List
from langchain_core.embeddings import Embeddings
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdfs(pdf_folder_path: str):
    documents = []
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder_path, filename))
            documents.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


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
#     model="qwen/qwen3-4b-2507", 
#     temperature=0.7
# )



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


def create_vectorstore_from_pdfs(pdf_folder_path: str, save_path: str = None):
    chunks = load_and_split_pdfs(pdf_folder_path)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    if save_path:
        vectorstore.save_local(save_path)

    return vectorstore


pdf_folder = "./Data"
save_path = "./faiss_index"
vectorstore = create_vectorstore_from_pdfs(pdf_folder, save_path=save_path)

print("Vector store created with", len(vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)), "embeddings.")

