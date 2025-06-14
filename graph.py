from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize LLM
try:
    llm = init_chat_model("google_genai:gemini-2.0-flash")
    print("LLM initialized successfully")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None

def chatbot(state: State):
    print("Entering chatbot function")
    print(f"State: {state}")
    
    if llm is None:
        raise Exception("LLM not initialized")
    
    try:
        system_prompt = SystemMessage(content="""You are a voice agent named Silica whose work is to solve the loneliness of people by talking to them and understanding them. Ask them questions about themselves, console them, and act as if you are their friend listening to them.""")
        
        # Convert input messages to proper format
        formatted_messages = [system_prompt]
        
        for msg in state["messages"]:
            if isinstance(msg, dict):
                # Handle dictionary format
                if msg.get("role") == "user":
                    formatted_messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    formatted_messages.append(msg)  # Keep as is if it's already a message object
                else:
                    # Default to human message
                    formatted_messages.append(HumanMessage(content=str(msg.get("content", msg))))
            else:
                # Handle message objects
                formatted_messages.append(msg)
        
        print(f"Formatted messages: {formatted_messages}")
        
        # Invoke the LLM
        response = llm.invoke(formatted_messages)
        print(f"LLM response: {response}")
        
        return {"messages": [response]}
        
    except Exception as e:
        print(f"Error in chatbot function: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_chat_graph(checkpointer=None):
    print("Creating chat graph...")
    
    try:
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        
        if checkpointer:
            graph = graph_builder.compile(checkpointer=checkpointer)
        else:
            graph = graph_builder.compile()
        
        print("Graph compiled successfully")
        return graph
        
    except Exception as e:
        print(f"Error creating graph: {e}")
        import traceback
        traceback.print_exc()
        raise
    
