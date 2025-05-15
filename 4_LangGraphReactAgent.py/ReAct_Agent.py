from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticToolsParser
from langgraph.graph import StateGraph,MessageGraph
from dotenv import load_dotenv
from langchain.agents import tool,create_react_agent
from langchain_community.tools import TavilySearchResults
from langchain import hub

import datetime
import operator
from typing import Annotated, TypedDict, Union

from langchain_core.agents import AgentFinish, AgentAction
from langgraph.graph import END, StateGraph


load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time
search_tool = TavilySearchResults(max_results=5)
react_prompt = hub.pull("hwchase17/react")

tools =[get_system_time,search_tool]

react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)

class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
#NODE

def reason_node(state: AgentState):
    agent_outcome = react_agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}


def act_node(state: AgentState):
    agent_action = state["agent_outcome"]
    
    # Extract tool name and input from AgentAction
    tool_name = agent_action.tool
    tool_input = agent_action.tool_input
    
    # Find the matching tool function
    tool_function = None
    for tool in tools:
        if tool.name == tool_name:
            tool_function = tool
            break
    
    # Execute the tool with the input
    if tool_function:
        if isinstance(tool_input, dict):
            output = tool_function.invoke(**tool_input)
        else:
            output = tool_function.invoke(tool_input)
    else:
        output = f"Tool '{tool_name}' not found"
    
    return {"intermediate_steps": [(agent_action, str(output))]}





REASON_NODE = "reason_node"
ACT_NODE = "act_node"

def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    else:
        return ACT_NODE


graph = StateGraph(AgentState)

graph.add_node(REASON_NODE, reason_node)
graph.set_entry_point(REASON_NODE)
graph.add_node(ACT_NODE, act_node)


graph.add_conditional_edges(
    REASON_NODE,
    should_continue,
)

graph.add_edge(ACT_NODE, REASON_NODE)

app = graph.compile()

result = app.invoke(
    {
        "input": "How many days ago was the latest SpaceX launch?", 
        "agent_outcome": None, 
        "intermediate_steps": []
    }
)

#print(result["agent_outcome"].return_values["output"], "final result")

print(result["agent_outcome"])
