import os
import subprocess
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from typing import List

def print_conversation(messages: List[BaseMessage]):
  for message in messages:
    message.pretty_print()

def custom_print_conversation(messages: List[BaseMessage]):
    for message in messages:
        # 1. Determine clean headers for standard visual blocks
        if isinstance(message, HumanMessage):
            header = "Human Message"
        elif isinstance(message, AIMessage):
            header = "Ai Message"
        elif isinstance(message, ToolMessage):
            header = "Tool Message"
        else:
            header = type(message).__name__
            
        print(f"================================ {header} ================================")
        
        # 2. TARGET IDENTIFIED BLOCK: Handle modern list-of-blocks layout from Responses API
        if isinstance(message.content, list):
            for block in message.content:
                # Isolate, parse, and print ONLY text response strings
                if isinstance(block, dict) and block.get("type") == "text":
                    print(block.get("text"))
                    
        # 3. FALLBACK BLOCK: Handle classic string message objects (like Human and Tool text)
        else:
            print(message.content)
            
        print("\n")
      
      
def display_graph(runnable: Runnable, output_png: Path) -> None:
  graph = runnable.get_graph()
  with output_png.open(mode="wb") as file:
    file.write(graph.draw_mermaid_png())
  
  print(f"Graph image successfully saved to local directory: {output_png.resolve()}")
  
  try:
    os.startfile(output_png)
  except AttributeError:
    # Fallback security handling for macOS/Linux systems if needed
    subprocess.run(["open", str(output_png)] if os.uname().sysname == "Darwin" else ["xdg-open", str(output_png)])