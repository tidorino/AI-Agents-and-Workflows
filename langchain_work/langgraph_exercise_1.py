import os

from langchain.agents import AgentState
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime
from pathlib import Path
from pydantic import SecretStr
from typing import List, TypedDict

from utils import custom_print_conversation, print_conversation, display_graph
from dotenv import load_dotenv
dotenv_path = Path(r'../env/.env.prod')
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.environ.get("OPENAI_API_KEY")


class CustomAgentState(AgentState):
  model_calls: int

class CustomAgentContext(TypedDict):
  user_id: str
  
  
@tool
def weather(city: str, runtime: ToolRuntime[CustomAgentContext]) -> str:
    """Return a (fake) current-weather report for a city."""
    print(f"Weather tool call executing for user_id: {runtime.context['user_id']}")
    namespace = ("users", runtime.context['user_id'], "recent_activity")
    runtime.store.put(namespace, "weather_requests", {"city": city})
    
    data = {
        "sofia": "Sofia: 18 C, partly cloudy",
        "london": "London: 11 C, rainy",
        "tokyo": "Tokyo: 22 C, sunny",
    }
    return data.get(city.lower(), f"No data for {city}.")

@tool
def search(query: str, runtime: ToolRuntime[CustomAgentContext]) -> str:
    """Look up a term in the built-in mini-encyclopedia."""
    print(f"Search tool call executing for user_id: {runtime.context['user_id']}")
    namespace = ("users", runtime.context['user_id'], "recent_activity")
    runtime.store.put(namespace, "search_requests", {"query": query})
    
    data = {
        "langgraph": "LangGraph is a library for building stateful, cyclic LLM apps.",
        "react": "ReAct is a prompting pattern: Reason then Act, in a loop.",
        "dag": "A DAG is a directed acyclic graph — no cycles allowed.",
    }
    return data.get(query.lower(), "(nothing found)")


SYSTEM_PROMPT = "You are a concise assistant. Use tools when useful."
TOOLS = [weather, search]

my_model = ChatOpenAI(model="gpt-5.6-luna", api_key=openai_api_key, reasoning_effort="none").bind_tools(TOOLS)
    
def on_start(state: CustomAgentState):
  print("Our graph execution started!")

def on_end(state: CustomAgentState):
  print("Our graph execution just ended!")

def before_model_node(state: CustomAgentState):
  model_call_id = state.get('model_calls', 0) + 1
  print('=' * 20)
  print(f"Starting model call #{model_call_id}...")
  return {"model_calls": model_call_id}

def after_model_node(state: CustomAgentState):
  model_call_id = state.get('model_calls', 0)
  print(f"Finished model call #{model_call_id}.")
  print('=' * 20)

def my_model_node(state: CustomAgentState):
  response = my_model.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
  return {"messages": [response]}

def has_pending_tool_calls(state: CustomAgentState) -> bool:
  messages = state.get("messages", [])
  if not messages:
    return False
  
  last_message = messages[-1]
  return isinstance(last_message, AIMessage) and last_message.tool_calls
  
graph_builder = StateGraph(CustomAgentState, CustomAgentContext)
graph_builder.add_node("on_start", on_start)
graph_builder.add_node("on_end", on_end)
graph_builder.add_node("before_model", before_model_node)
graph_builder.add_node("after_model", after_model_node)
graph_builder.add_node("model", my_model_node)
graph_builder.add_node("tools", ToolNode(TOOLS))

graph_builder.add_edge(START, "on_start")
graph_builder.add_edge("on_start", "before_model")
graph_builder.add_edge("before_model", "model")
graph_builder.add_edge("model", "after_model")
graph_builder.add_conditional_edges("after_model", lambda x: "tools" if has_pending_tool_calls(x) else "on_end", ["tools", "on_end"])
graph_builder.add_edge("tools", "model")
graph_builder.add_edge("on_end", END)
graph = graph_builder.compile(checkpointer=InMemorySaver(), store=InMemoryStore())

display_graph(graph, Path("./graph.png"))

final_state = graph.invoke(
    input={
        "messages": [HumanMessage("What's the weather in Tokyo and what is LangGraph, briefly?")]
    },
    config={ "configurable": { "thread_id": "thread_1" } },
  context={ "user_id": "user_1" }
)

print_conversation(final_state["messages"])

state_history = list(graph.get_state_history({"configurable": { "thread_id": "thread_1" }}))
for el in reversed(state_history):
  print(f'Step: {el.metadata["step"]}')
  print("Current state:")
  print(el.values)
  print(f"Next: {el.next}")
  print()