import os
from pathlib import Path
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import PIIMiddleware, ModelRequest, ModelResponse, ToolCallRequest
from langchain.agents.middleware import after_agent, after_model, before_agent, before_model, wrap_model_call, wrap_tool_call
from langchain.messages import HumanMessage, ToolMessage
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain.tools import tool
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
from pydantic import SecretStr
from typing import Callable

from utils import print_conversation,  display_graph, custom_print_conversation

from dotenv import load_dotenv

dotenv_path = Path(r'../env/.env.prod')
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.environ.get("OPENAI_API_KEY")

graph_file_path = Path("./retriever_workflow.png")


@tool
def get_interesting_fact() -> str:
    """
    This tool will discover an interesting fact to you.
    """
    return "The Earth is actually not a perfect sphere."

# NOTE: Middlewares can be implemented by separate functions or by a single class inheriting from `AgentMiddleware`.
@before_agent
def before_agent_func(state: AgentState, runtime: Runtime) -> None:
    print("Event: before_agent")
    print(state)
    print(runtime)

@before_model
def before_model_func(state: AgentState, runtime: Runtime) -> None:
    print("Event: before_model")
    print(state)
    print(runtime)

# @after_model
# def after_model_func(state: AgentState, runtime: Runtime) -> None:
#     print("Event: after_model")
#     print(state)
#     print(runtime)
@after_model
def after_model_func(state_dict: dict, runtime_obj: Runtime) -> None:
    print("\n=== Event: after_model ===")

    # 1. EXTRACT THE LATEST MESSAGE: Get the last item from the messages list
    messages = state_dict.get("messages", [])
    if not messages:
        print("⚠️ No messages found in agent state.")
        return

    output_message = messages[-1]

    # 2. VERIFY AND EXTRACT REASONING METRICS
    if isinstance(output_message, AIMessage) and hasattr(output_message, "response_metadata"):
        metadata = output_message.response_metadata
        token_details = metadata.get("token_usage", {}).get("completion_tokens_details", {})
        count = token_details.get("reasoning_tokens", 0)

        print(f"🧠 [Trace Metric] Internally consumed: {count} reasoning tokens.")

        # 3. Extract Text Thought Summary Blocks if present
        if isinstance(output_message.content, list):
            for block in output_message.content:
                if isinstance(block, dict) and block.get("type") == "reasoning":
                    print(f"🤔 [Thought Summary]: {block.get('text')}\n")
    else:
        print("⚠️ The latest message was not an AIMessage or lacks metadata.")

@after_agent
def after_agent_func(state: AgentState, runtime: Runtime) -> None:
    print("Event: after_agent")
    print(state)
    print(runtime)

@wrap_model_call
def handle_model_call(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]):
    print("Event: model_call")
    print(request)

    # Only force tool on the first LLM step
    if len(request.messages) == 1:
        print("Forcing a specific tool call")
        request = request.override(tool_choice=get_interesting_fact.name)

    response = handler(request)

    print("Obtained model response:")
    print(response)

    return response

@wrap_tool_call
def handle_tool_call(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage]):
    print("Event: tool_call")
    print(request)

    response = handler(request)

    print("Obtained tool message:")
    print(response)

    return response

agent = create_agent(
    model=ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=openai_api_key,
        use_responses_api=True,
        reasoning_effort="low"
    ),
    tools=[get_interesting_fact],
    middleware=[
        PIIMiddleware(pii_type="email", strategy="redact"),
        before_agent_func,
        before_model_func,
        after_agent_func,
        after_model_func,
        handle_model_call,
        handle_tool_call
    ]
)
display_graph(agent, graph_file_path)
reserve_ticket = agent.invoke(
    input={
        "messages": [
            HumanMessage("Share an intresting fact with me and then book two seats under maria.popova@example.com for the play at the theater this evening.")
        ]
    }
)

# print_conversation(reserve_ticket["messages"])
custom_print_conversation(reserve_ticket["messages"])