from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages, StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()


GENERATE_POST = "chat"
GET_REVIEW_DECISION = "user_check"
POST = "Post"
COLLECT_FEEDBACK = "feedback"


llm=ChatGoogleGenerativeAI(model='gemini-1.5-pro')

search_tool= TavilySearchResults(max_results=5)

tool=[search_tool]

llm_with_tools=llm.bind_tools(tools=tool)

prompt = ChatPromptTemplate.from_messages([
    ("human", "You are a helpful and expert AI assistant. Write a LinkedIn post on the topic: {topic}")
])

# User input
topic = input("Enter the Topic Name on which you want to post the blog: ")
prompt=prompt.invoke({'topic':topic})
class State(TypedDict): 
    messages: Annotated[list, add_messages]
    
def chat(state: State):
    response= {
        'messages':[llm.invoke(state['messages'])]
    }
    return response
def user_check(state: State):
    Output=state['messages'][-1].content
    print("\n📢 Current LinkedIn Post:\n")
    print(Output)
    print('\n')
    use_input= input("Check if it is Good According to You and Gave me The Answer in Yes or No : ")
    if use_input.lower() == 'yes':
        return POST
    else:
        return COLLECT_FEEDBACK
def Post(state: State):
    print("***---------------------------***")
    print("Post Uploaded Sucessfully")
    print("***---------------------------***")
def feedback(state: State):
    feedback=input('Please Gave me Feedback For Area Of Inprovement in the BLog : ')
    return{
        "messages":HumanMessage(content=feedback)
    }
graph=StateGraph(State)

graph.add_node(GENERATE_POST,chat)
graph.add_node(GET_REVIEW_DECISION,user_check)
graph.add_node(POST,Post)
graph.add_node(COLLECT_FEEDBACK,feedback)
graph.set_entry_point(GENERATE_POST)
graph.add_conditional_edges(GENERATE_POST,user_check)
graph.add_edge(POST,END)
graph.add_edge(COLLECT_FEEDBACK,GENERATE_POST)
app=graph.compile()
app.get_graph().print_ascii()
response=app.invoke(prompt)

print(response)