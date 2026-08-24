from pathlib import Path
import os

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.messages import HumanMessage, ToolMessage, SystemMessage
from langchain.tools import tool
from langchain_core.messages import BaseMessage

from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


dotenv_path = Path(r'../env/.env.prod')
load_dotenv(dotenv_path=dotenv_path)

gemini_api_key = os.environ.get("GEMINI_API_KEY")


def print_conversation(messages: List[BaseMessage]):
  for message in messages:
    message.pretty_print()

@tool
def get_database_status():
  """
  Returns information about the current database status.
  """
  return "healthy"

tools = [get_database_status]
tools_registry = { t.name: t for t in tools }
openai_api_key = os.environ.get("OPENAI_API_KEY")
gemini_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
model_with_tools = gemini_model.bind_tools(tools)
openai_default_model = ChatOpenAI(model="gpt-5.6-luna",
    reasoning_effort="none", api_key=openai_api_key).bind_tools(tools)


def run_agent_loop(conversation: List[BaseMessage]):
  MAX_ITERATIONS = 100
  
  for i in range(MAX_ITERATIONS):
    
    reply = openai_default_model.invoke(conversation)
    conversation.append(reply)
    
    # No tool calls = agent has finished
    if not reply.tool_calls:
      return conversation
    
    # Execute requested tools
    for tool_call in reply.tool_calls:
      tool_call_id = tool_call["id"]
      tool_call_name = tool_call["name"]
      tool_call_args = tool_call["args"]
      
      tool_result = tools_registry[tool_call_name].invoke(
        tool_call_args
      )
      
      conversation.append(
        ToolMessage(
          content=str(tool_result),
          tool_call_id=tool_call_id
        )
      )
  

  raise RuntimeError(
    f"Could not finish the interaction within {MAX_ITERATIONS} iterations."
  )

messages = [
    SystemMessage("You are a helpful customer support agent."),
    HumanMessage("What is current status of my database?"),
]
final_conversation = run_agent_loop(messages)
print_conversation(final_conversation)