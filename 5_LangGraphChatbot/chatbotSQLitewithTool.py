from typing import TypedDict, Annotated, List
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

llm = ChatGroq(model="llama-3.1-8b-instant")

class BasicChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chatbot(state: BasicChatState):
    return {
        "messages": [llm.invoke(state["messages"])]
    }

graph = StateGraph(BasicChatState)
graph.add_node("chatbot", chatbot)
graph.add_edge("chatbot", END)
graph.set_entry_point("chatbot")

app = graph.compile(checkpointer=memory)

config = {"configurable": {
    "thread_id": "1"
}}

while True:
    user_input = input("User: ").strip()
    if user_input.lower() in ["exit", "end", "quit"]:
        break
    result = app.invoke({
        "messages": [HumanMessage(content=user_input)]
    }, config=config)
    
    response = result.get("messages", [])
    if response:
        print("AI:", response[-1].content)
    else:
        print("AI: [No response]")
