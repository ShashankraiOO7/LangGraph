from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

parser = StrOutputParser()

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot( state: BasicChatState):
    return {
        "messages": [llm.invoke(state["messages"])]
    }

graph=StateGraph(BasicChatState)

graph.add_node("chatbot",chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot",END)

app=graph.compile()

while True: 
    user_input = input("User: ")
    if(user_input in ["exit", "end"]):
        break
    else: 
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

        print(result.content)

        

