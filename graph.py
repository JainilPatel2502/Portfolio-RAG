from langgraph.graph import StateGraph, END
from schemas.rag_state import State
from nodes.retrieve import retrieve
from nodes.generate import generate
workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()