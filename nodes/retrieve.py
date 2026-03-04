from langchain_community.vectorstores import FAISS
from schemas.rag_state import State
import os
from llms.model import embeddings

# if os.path.exists("./faiss_index_local"):
#     db = FAISS.load_local(
#         "./faiss_index_local",
#         embeddings,
#         allow_dangerous_deserialization=True
#     )
#     retriever = db.as_retriever(search_kwargs={"k": 3})
# else:
#     print("Warning: ./faiss_index not found. Retrieval will fail.")
#     retriever = None

if os.path.exists("./faiss_index"):
    db = FAISS.load_local(
        "./faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 7})
else:
    print("Warning: ./faiss_index not found. Retrieval will fail.")
    retriever = None

def retrieve(state: State):
    messages = state["human_message"]
    last_message = messages[-1]["content"]
    if retriever:
        docs = retriever.invoke(last_message)
        contents = "\n\n".join([doc.page_content for doc in docs])
    else:
        contents = ""
    return {"context": contents}