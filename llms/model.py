from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from typing import List, Dict, Any, TypedDict, Annotated , Optional
import requests
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
