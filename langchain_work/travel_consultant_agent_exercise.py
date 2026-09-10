import json
import os
from pathlib import Path
from typing import TypedDict

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, HumanInTheLoopMiddleware
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.pregel.main import Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from utils import custom_print_conversation
from dotenv import load_dotenv
dotenv_path = Path(r'../env/.env.prod')
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.environ.get("OPENAI_API_KEY")

my_checkpointer = InMemorySaver()
my_store = InMemoryStore()
my_model = ChatOpenAI(model="gpt-5.6-luna", api_key=openai_api_key, use_responses_api=True, reasoning_effort="none")
config_1 = {"configurable": {"thread_id": "thread_8"}}
context_1 = {"user_id": "first_user_3"}

class TravelConsultantAgentContext(TypedDict):
    user_id: str

# @before_model
# def before_model_func(state: AgentState, runtime: Runtime[TravelConsultantAgentContext]) -> None:
#     print(f"Event: before_model; User ID: {runtime.context.get('user_id')}")

@tool
def read_preferences(runtime: ToolRuntime[TravelConsultantAgentContext]) -> str:
    """
    Read user preferences . Call at the start of every session.
    Example: "dietary: vegetarian; seat_preference: aisle"
    """
    # Here you would implement the logic to read the preferences from a database or memory store.
    # For demonstration, we will just return a mock preference string.
    preferences_namespace = ("users", runtime.context["user_id"], "preferences")
    preferences = runtime.store.search(preferences_namespace, limit=100)
    
    if not preferences:
        return "No preferences found."
    
    preference_strings = [f"{key}: {value['preference']}" for key, value in preferences.items()]
    return "; ".join(preference_strings)


@tool
def save_preference(preference_key: str, preference_value: str, runtime: ToolRuntime[TravelConsultantAgentContext]) -> str:
    """
    Save user preferences in long-term memory.
    Example: "preference_key: destination; preference_value: Japan"
    """
    # Here you would implement the logic to save the preference in a database or memory store.
    # For demonstration, we will just return a confirmation message.
    runtime.store.put(
      namespace=('preferences', runtime.context.get('user_id')),
      key=preference_key,
      value={ "preference": preference_value})
    
    return f"Preference saved: {preference_key} = {preference_value}"

@tool
def book_travel(destination:str, nights: int) -> str:
  """ Call this tool to book a trip on behalf of the user. Returns a confirmation message. """
  return f"Your trip to {destination} for {nights} nights has been successfully booked. "


DESTINATIONS = {
    "amalfi": { "name": "Amalfi Coast, Italy", "season": "May-Sep", "from_eur": 18_500 },
    "bora": { "name": "Bora Bora, French Polynesia", "season": "Apr-Oct", "from_eur": 32_000 },
    "kyoto": { "name": "Kyoto, Japan", "season": "Mar-May, Oct-Nov", "from_eur": 24_000 },
    "aspen": { "name": "Aspen, Colorado", "season": "Dec-Mar", "from_eur": 21_500 },
    "marrakech": { "name": "Marrakech, Morocco", "season": "Oct-Apr", "from_eur": 15_750 },
}

CATALOGUE = {
    "destinations": { key: dest["name"] for key, dest in DESTINATIONS.items() },
    "services": [
        "Private-jet transfers via NetJets and VistaJet",
        "Forbes 5-Star and Relais & Chateaux properties only",
        "Personal butler and dedicated 24/7 destination manager",
        "Michelin-experience curation (typically 2-3* venues)",
    ]
}

CATALOGUE_AS_CONTEXT = json.dumps(CATALOGUE, indent=2)

SYSTEM_PROMPT = f"""You are **Jacque**, the AI concierge of an ultra-luxury travel atelier.
You speak with warm restraint - think a Parisian maitre d'hotel, never a salesperson.

# Catalogue you may sell from
{CATALOGUE_AS_CONTEXT}

# Operating rules
1.  ALWAYS call `{read_preferences.name}` as your very first action in a new session, then weave the
   user's known preferences into your reply so they feel recognised.
2. When the user reveals a durable preference that was not previously recalled (allergy, favourite activities, etc.),
   call `{save_preference.name}`. Use stable, lowercase keys (e.g. `dietary`, `seat_preference`, `favourite_hotel_brand`).
3. When the user is ready to book, call `{book_travel.name}` with the destination and number of nights.

# Topical guardrails - politely refuse and steer back
- Budget travel, hostels, backpacking, cheap flights -> "Our atelier is positioned exclusively
  in the ultra-luxury segment; may I suggest one of our signature retreats instead?"
- Competing agencies (Abercrombie & Kent, Black Tomato, etc.) -> decline to compare; redirect.
- Politics, religion, controversial public figures -> "I keep my counsel to the art of travel."
- Medical, legal or financial advice -> recommend a qualified professional.
"""


travel_agent = create_agent(
  model=my_model,
  tools=[save_preference, read_preferences, book_travel],
  checkpointer=my_checkpointer,
  store=my_store,
  context_schema=TravelConsultantAgentContext,
  middleware=[HumanInTheLoopMiddleware(interrupt_on={book_travel.name: True})],
  system_prompt=SYSTEM_PROMPT,
  debug=True
)

response = travel_agent.invoke(
  input={"messages":[
    HumanMessage("Hello! My name is Smith .")
  ]
  },
  config=config_1,
  context=context_1
)

response_1 = travel_agent.invoke(
  input={"messages":[
    HumanMessage("I'm planning an anniversary trip in late May for my wife and me. "
                "I'm vegetarian. What can you recommend?")
  ]
  },
  config=config_1,
  context=context_1
)

custom_print_conversation(response_1["messages"])

response_2 = travel_agent.invoke(
  input={"messages":[
    HumanMessage("I'm planning a trip for 7 nights. ")
  ]
  },
  config=config_1,
  context=context_1
)

response_3 = travel_agent.invoke(
  input=Command(resume={"allowed_decisions": [{"type": "reject"}]}),
  config=config_1,
  context=context_1
)

# for namespace in my_store.list_namespaces():
#   res = my_store.search(namespace=namespace)
#   print(res)
#   my_store.get(namespace=namespace, key=None)
  
custom_print_conversation(response_3["messages"])

for x in my_checkpointer.list(config=config_1):
    print(f"Checkpoint: {x}")
