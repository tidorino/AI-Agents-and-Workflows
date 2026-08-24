from pathlib import Path
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from typing import List

from dotenv import load_dotenv

dotenv_path = Path(r'../env/.env.prod')
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.environ.get("OPENAI_API_KEY")
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
"""
if use openai_model:
openai_model = ChatOpenAI(model="gpt-5.6-luna", api_key=openai_api_key).bind_tools(tools)
"""
gemini_model = ChatGoogleGenerativeAI( model="gemini-3.6-flash")

agent = create_agent(
    model=gemini_model,
    tools=tools,
    system_prompt="You are a helpful customer support agent. ",
    debug=False
  )

result = agent.invoke({"messages": [HumanMessage("What is current status of my database?")]})
print_conversation(result['messages'])
print(f'================================== Ai Message ==================================\n{result["messages"][-1].text}')
