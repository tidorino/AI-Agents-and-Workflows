import os
import sqlite3
import pandas as pd
from pathlib import Path
from typing import TypedDict


from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import PIIMiddleware, ModelRequest, ModelResponse, ToolCallRequest
from langchain.agents.middleware import after_agent, after_model, before_agent, before_model, wrap_model_call, wrap_tool_call
from langchain.messages import HumanMessage, ToolMessage
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda
from langchain.tools import tool
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from utils import print_conversation,  display_graph, custom_print_conversation

from dotenv import load_dotenv

dotenv_path = Path(r'../env/.env.prod')
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.environ.get("OPENAI_API_KEY")

graph_file_path = Path("./retriever_workflow.png")

class CustomAgentContext(TypedDict):
  user_id: str

def explore_database(connection: sqlite3.Connection, table_name):
  result_df = pd.read_sql_query(
    f'SELECT * FROM "{table_name}"',
    connection
  )
  
  display_graph(f"## {table_name}")
  display_graph(result_df)

checkpointer_connection = sqlite3.connect("/content/checkpointer.db", check_same_thread=False, isolation_level=None)
checkpointer = SqliteSaver(checkpointer_connection)
checkpointer.setup()

store_connection = sqlite3.connect("/content/store.db", check_same_thread=False, isolation_level=None)
store = SqliteStore(store_connection)
store.setup()

@tool
def remember_user_facts(key: str, value: str, runtime: ToolRuntime[CustomAgentContext]) -> str:
  """
  Extract durable user facts from a user message and store them in long-term memory.
  Example: "key: allergy; value: The user is allergic to nuts.", "key: hobbies; value: The user can play the guitar.|
  Args:
    key: A unique identifier of the fact.
    value: The fact itself.
  """
  
  namespace = ("user",  runtime.context["user_id"], "general_knowledge")
  prev_item = runtime.store.get(namespace, "auto_extracted_facts")
  facts_dict = prev_item.value if prev_item is not None else {}
  facts_dict[key] = value
  runtime.store.put(namespace, "auto_extracted_facts", facts_dict)
  return "OK"

@tool
def recall_user_facts(runtime: ToolRuntime[CustomAgentContext]) -> str:
  """
  Recall previously stored long-term facts about the user.
  """
  namespace = ("user", runtime.context["user_id"], "general_knowledge")
  results = runtime.store.search(namespace, limit=20)
  if not results:
    return "No facts stored."
  return '\n===\n'.join(f"{facts_group.key}:\n{'\n'.join(f' - {key}: \"{value}\"' for key, value in facts_group.value.items())}" for facts_group in results)

@tool
def get_interesting_fact() -> str:
    """
    This tool will discover an interesting fact to you.
    """
    return "The Earth is actually not a perfect sphere."



agent = create_agent(
    model=ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=openai_api_key,
        use_responses_api=True,
        reasoning_effort="low"
    ),
    tools=[remember_user_facts, recall_user_facts],
    system_prompt=f""" You are polite and helpful personal assistant. You should use frequently the \"{remember_user_facts.name}\" tool to store information about
                                the user that can be useful in future.
                                At the start of each iteration, ALWAYS use the \"{remember_user_facts.name}\" tool.
                                Be freandly - include known facts in the conversation to make the user feel special.
                                Proactively store useful data about the users - name, hobbies, plans, needs, ect.
                                """,
    checkpointer=checkpointer,
    store=store,
    context_schema=CustomAgentContext,
    debug=True
)

interact = agent | RunnableLambda(lambda res: print_conversation(res["messages"]))

res = interact.invoke(
  input={
    "messages": [
      HumanMessage("Hello, I would like to help me with the management of my personal notes and timeline.")
    ]
  },
  config={
    "configurable": {"thread_id": "thread_1"}
  },
  context={"user_id": "thread"}
)
custom_print_conversation(res["messages"])

explore_database(checkpointer_connection, "checkpoints")
explore_database(store_connection, "store")

res1 = interact.invoke(
  input={
    "messages": [
      HumanMessage("I want to plan a business meeting for today, 15:00.")
    ]
  },
  config={
    "configurable": {"thread_id": "thread_1"}
  },
  context={"user_id": "thread"}
)
custom_print_conversation(res1["messages"])
# result = agent.invoke(
#     input={
#         "messages": [
#             HumanMessage("Share an intresting fact with me.")
#         ]
#     },
#   config={
#     "configurable": { "thread_id": "thread_1"}
#   }
# )
# custom_print_conversation(result["messages"])
# result2 =  agent.invoke(
#     input={
#         "messages": [
#             HumanMessage("Share an intresting fact with me.")
#         ]
#     },
#   config={
#     "configurable": { "thread_id": "thread_1"}
#   }
# )
# custom_print_conversation(result2["messages"])