from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

memory = MemorySaver()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

class BasicChatState(TypedDict): 
    messages: Annotated[list, add_messages]

def chatbot(state: BasicChatState): 
    return {
       "messages": [llm.invoke(state["messages"])]
    }

graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)

graph.add_edge("chatbot", END)

graph.set_entry_point("chatbot")

app = graph.compile(checkpointer = memory)

config = {"configurable": {
    "thread_id": 1
}}

while True: 
    user_input = input("User: ")
    if(user_input in ["exit", "end"]):
        break
    else: 
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        }, config=config)

        print("AI: " + result["messages"][-1].content)