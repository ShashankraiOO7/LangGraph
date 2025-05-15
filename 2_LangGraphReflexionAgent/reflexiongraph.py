
from typing import List

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chain import revisor_chain, first_responder_chain
from executetool import execute_tools

graph = MessageGraph()
MAX_ITERATIONS = 3

graph.add_node("draft", first_responder_chain)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor_chain)


graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")

def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

graph.add_conditional_edges("revisor", event_loop)
graph.set_entry_point("draft")

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()
response = app.invoke(
    "Write about how small business can leverage AI to grow"
)

print(response[-1].tool_calls[0]["args"]["answer"])
print(response, "response")

from IPython.display import display, Image

# Display Mermaid diagram (PNG)
display(Image(app.get_graph().draw_mermaid_png()))

