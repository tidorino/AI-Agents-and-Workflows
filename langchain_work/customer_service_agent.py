import json
import os
from pathlib import Path

from langchain.agents import create_agent, AgentState
from langgraph.runtime import Runtime
from langchain.messages import HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain.tools import tool
from langchain.agents.middleware import after_model,before_model
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from utils import custom_print_conversation, print_conversation, display_graph

from dotenv import load_dotenv

dotenv_path = Path(r'../env/.env.prod')
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.environ.get("OPENAI_API_KEY")
chroma_store = Chroma(
    collection_name="faq",
    persist_directory="./chroma",
)

text_loader = TextLoader(file_path=Path("./FQA.md"), encoding="utf-8")
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('##', 'question')], strip_headers=False)

documents = text_loader.load()
chunks = []
for doc in documents:
    chunks.extend(splitter.split_text(doc.page_content))

chroma_store.add_documents(chunks)
chroma_retriever = chroma_store.as_retriever(search_kwargs = {"k": 3})
search_knowledgebase = create_retriever_tool(chroma_retriever, "search_knowledgebase", " Call this tool to search the internal knowledgebase using a natural language query.", document_separator="\n\n----\n\n")

@tool
def my_orders() -> str:
    """
    Call this tool to retrieve all orders of the customer.
    """
    
    orders = [{"id": 13, "status": "pending"}, {"id": 27, "status": "delivered"}]
    return json.dumps(orders)

@tool
def lookup_order_details(order_id: int):
    """
    Call this tool to lookup additional information about a given order by its id.
    """
    details = {
        13: { "price": 3.14, "vat": 1.06, "products": "whole milk"},
        27: { "price": 17.89, "vat": 3.66, "products": "keyboard"},
    }
    results = details.get(order_id)
    if  results is None:
        return "An order with this id was not found"
    
    return json.dumps(results)

# @tool
# def search_knowledgebase(query: str):
#     """
#     Call this tool to search the internal knowledgebase using a natural language query.
#     :param query:
#     :return:
#     """
#     results = chroma_store.similarity_search(query, k=3)
#     return "\n\n----\n\n".join(d.page_content for d in results)
# Test only one tool
# result = lookup_order_details.invoke({"order_id": 13})
# print(result)
@before_model
def before_model_func(state: AgentState, runtime: Runtime) -> None:
    print("Event: before_model")
    
@after_model
def after_model_func(state: AgentState, runtime: Runtime) -> None:
    print(f"Usage: { state["messages"][-1].usage_metadata}")

agent = create_agent(
    model=ChatOpenAI(model="gpt-5.6-luna", api_key=openai_api_key, use_responses_api=True),
    tools=[search_knowledgebase, my_orders, lookup_order_details],
    middleware=[after_model_func, before_model_func],
    system_prompt="You are helpful customer support agent.",
    debug=True
)

display_graph(agent, Path("./graph.png"))
# exploration_response = agent.invoke(input={"messages": [HumanMessage("Hello ! How long should I wait for a standart delivery? Do you offer an express option?")]})
my_orders_response = agent.invoke(input = {"messages": [HumanMessage("Can you give me a detailed information about my orders?")]})
# custom_print_conversation(exploration_response["messages"])
custom_print_conversation(my_orders_response["messages"])
