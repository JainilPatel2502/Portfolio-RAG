from typing import TypedDict, List, Annotated
from langchain_core.messages import AIMessage, HumanMessage
import operator

class State(TypedDict):
    ai_messages: Annotated[List[AIMessage], operator.add]
    human_message: Annotated[List[HumanMessage], operator.add]
    context: str
    