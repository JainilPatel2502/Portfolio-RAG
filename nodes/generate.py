from itertools import zip_longest
from schemas.rag_state import State
from llms.model import llm
from prompts.rag_prompt import prompt

def generate(state: State):
    human_msgs = state["human_message"]
    ai_msgs = state["ai_messages"]
    context = state["context"]
    history = ""
    
    # Get last 3 messages or all if less than 3
    recent_human = human_msgs[-3:] if len(human_msgs) >= 3 else human_msgs
    recent_ai = ai_msgs[-3:] if len(ai_msgs) >= 3 else ai_msgs
    
    # Only build history if there are previous messages
    if len(recent_human) > 1 or len(recent_ai) > 0:
        for h, a in zip_longest(recent_human[:-1], recent_ai, fillvalue=None):
            if h:
                if isinstance(h, dict):
                    history += "Human: " + str(h["content"]) + "\n"
                else:
                    history += "Human: " + str(h.content) + "\n"
            if a:
                if isinstance(a, dict):
                    history += "AI: " + str(a["content"]) + "\n"
                else:
                    history += "AI: " + str(a.content) + "\n"
    
    # Get current question
    last_human = human_msgs[-1]
    if isinstance(last_human, dict):
        question = last_human["content"]
    else:
        question = last_human.content
    
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "history": history,
        "question": question
    })
    return {"ai_messages": [response]}
