from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import json
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_community.tools import TavilySearchResults
load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# ✅ FIXED: use from_messages
actor_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are expert AI researcher.
1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection."""
    ),
    MessagesPlaceholder(variable_name="messages"),
    (
        "system",
        "Answer the user's question above using the required format."
    )
])

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    """Answer the question."""
    answer: str = Field(description="~250 word detailed answer to the question.")
    search_queries: List[str] = Field(description="1-3 search queries for researching improvements.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
    


revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""
first_responder_chain = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
) | llm.bind_tools(tools=[AnswerQuestion], tool_choice='AnswerQuestion')

revise_responder_chain= actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


print("---------------------------------------------")

print(first_responder_chain)
print("---------------------------------------------")
print(revise_responder_chain)
print("---------------------------------------------"),print("---------------------------------------------"),print("---------------------------------------------")



# Create the Tavily search tool
tavily_tool = TavilySearchResults(max_results=5)

def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    last_ai_message: AIMessage = state[-1]
    
    # Extract tool calls from the AI message
    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []
    
    # Process the AnswerQuestion or ReviseAnswer tool calls to extract search queries
    tool_messages = []
    
    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])
            
            # Execute each search query using the tavily tool
            query_results = {}
            for query in search_queries:
                result = tavily_tool.invoke(query)
                query_results[query] = result
            
            # Create a tool message with the results
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(query_results),
                    tool_call_id=call_id
                )
            )
    
    return tool_messages

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
