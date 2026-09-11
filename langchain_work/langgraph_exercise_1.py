import os

from langchain.agents import AgentState
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pathlib import Path
from pydantic import SecretStr
from typing import List

from utils import custom_print_conversation, print_conversation, display_graph
from dotenv import load_dotenv
dotenv_path = Path(r'../env/.env.prod')
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.environ.get("OPENAI_API_KEY")

@tool
def weather(city: str) -> str:
    """Return a (fake) current-weather report for a city."""
    data = {
        "sofia": "Sofia: 18 C, partly cloudy",
        "london": "London: 11 C, rainy",
        "tokyo": "Tokyo: 22 C, sunny",
    }
    return data.get(city.lower(), f"No data for {city}.")

@tool
def search(query: str) -> str:
    """Look up a term in the built-in mini-encyclopedia."""
    data = {
        "langgraph": "LangGraph is a library for building stateful, cyclic LLM apps.",
        "react": "ReAct is a prompting pattern: Reason then Act, in a loop.",
        "dag": "A DAG is a directed acyclic graph — no cycles allowed.",
    }
    return data.get(query.lower(), "(nothing found)")


SYSTEM_PROMPT = "You are a concise assistant. Use tools when useful."
TOOLS = [weather, search]

my_model = ChatOpenAI(model="gpt-5.6-luna", api_key=openai_api_key, reasoning_effort="none").bind_tools(TOOLS)

class CustomAgentState(AgentState):
    model_calls: int


def model_node(state: CustomAgentState):
  response = my_model.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
  return {"messages": [response]}


def has_pending_tool_calls(state: CustomAgentState) -> bool:
  messages = state.get("messages", [])
  if not messages:
    return False
  
  last_message = messages[-1]
  return isinstance(last_message, AIMessage) and last_message.tool_calls
  
graph_builder = StateGraph(CustomAgentState)
graph_builder.add_node("model", model_node)
graph_builder.add_node("tools", ToolNode(TOOLS))

graph_builder.add_edge(START, "model")
graph_builder.add_conditional_edges("model", lambda x: "tools" if has_pending_tool_calls(x) else END, ["tools", END])
graph_builder.add_edge("tools", "model")
graph = graph_builder.compile()

display_graph(graph, Path("./graph.png"))

final_state = graph.invoke(
    input={
        "messages": [HumanMessage("What's the weather in Tokyo and what is LangGraph, briefly?")]
    }
)

print_conversation(final_state["messages"])