from langchain_core.messages import HumanMessage,AIMessage
from dotenv import load_dotenv
from typing import List,Sequence
from langgraph.graph import END,MessageGraph
from chain import generation_chain,reflection_chain

load_dotenv()

REFLECT='reflect'
GENERATE='generate'
graph = MessageGraph()


def generate_node(state):
    response = generation_chain.invoke({
        'messages': state
    })
    return [AIMessage(content=response)]

def reflect_node(messages):
    response= reflection_chain.invoke({
        'messages': messages
    })
    return [HumanMessage(content=response)]

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)
graph.set_entry_point(GENERATE)

def should_continue(state):
    if (len(state) > 3):
        return END 
    return REFLECT


graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke(HumanMessage(content="AI Agents taking over content creation"))

print(response)