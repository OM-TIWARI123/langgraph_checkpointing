from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
from graph import create_chat_graph
import io
import wave
import tempfile
import os
from dotenv import load_dotenv
import traceback

load_dotenv()

app = FastAPI(title="Silica Voice Agent API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB configuration
MONGODB_URI = "mongodb://admin:admin@localhost:27017"

# Global checkpointer and graph
checkpointer = None
graph = None

def get_or_create_checkpointer():
    """Get or create a fresh MongoDB checkpointer"""
    global checkpointer
    try:
        # Always create a fresh connection for each request
        return MongoDBSaver.from_conn_string(MONGODB_URI)
    except Exception as e:
        print(f"Error creating checkpointer: {e}")
        traceback.print_exc()
        return None

@app.on_event("startup")
async def startup_event():
    global graph
    try:
        # Create graph without checkpointer initially for testing
        print("Creating graph without persistent checkpointer for now...")
        graph = create_chat_graph(checkpointer=None)
        print("Graph created successfully:", graph)
    except Exception as e:
        print(f"Error during startup: {e}")
        traceback.print_exc()

@app.on_event("shutdown")
async def shutdown_event():
    # Clean shutdown - checkpointers are created per request now
    print("Shutting down application...")

# Pydantic models for request/response
class TextChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    thread_id: str

class AudioChatRequest(BaseModel):
    thread_id: Optional[str] = "default"

# Text-based chat endpoint
@app.post("/chat/text", response_model=ChatResponse)
async def chat_text(request: TextChatRequest):
    """
    Send a text message to Silica and get a text response
    """
    try:
        print("Entering chat/text endpoint")
        print(f"Request: {request}")
        
        if graph is None:
            raise HTTPException(status_code=500, detail="Graph not initialized")
        
        # For now, let's test without persistent storage
        # You can enable MongoDB persistence later once basic functionality works
        config = {"configurable": {"thread_id": request.thread_id}}
        print(f"Config: {config}")
        
        # Create the input message in the correct format
        input_data = {
            "messages": [{"role": "user", "content": request.message}]
        }
        print(f"Input data: {input_data}")
        
        print("Starting graph stream...")
        response_message = None
        
        # Stream the response from the graph
        try:
            # Since we're not using persistent checkpointer, config can be simpler
            for event in graph.stream(input_data, stream_mode="values"):
                print(f"Event: {event}")
                if "messages" in event and event["messages"]:
                    response_message = event["messages"][-1]
                    print(f"Response message: {response_message}")
        except Exception as stream_error:
            print(f"Error during streaming: {stream_error}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Streaming error: {str(stream_error)}")
        
        if response_message is None:
            raise HTTPException(status_code=500, detail="No response generated")
        
        # Handle different message types
        response_content = ""
        if hasattr(response_message, 'content'):
            response_content = response_message.content
        elif isinstance(response_message, dict) and 'content' in response_message:
            response_content = response_message['content']
        else:
            response_content = str(response_message)
        
        return ChatResponse(
            response=response_content,
            thread_id=request.thread_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Audio-based chat endpoint
@app.post("/chat/audio", response_model=ChatResponse)
async def chat_audio(
    audio_file: UploadFile = File(...),
    thread_id: Optional[str] = "default"
):
    """
    Send an audio file to Silica and get a text response
    """
    try:
        # Validate file type
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read the audio file
        audio_data = await audio_file.read()
        
        # Initialize speech recognizer
        r = sr.Recognizer()
        
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Process the audio file
            with sr.AudioFile(temp_file_path) as source:
                audio = r.record(source)
            
            # Convert speech to text
            try:
                text = r.recognize_google(audio)
            except sr.UnknownValueError:
                raise HTTPException(status_code=400, detail="Could not understand audio")
            except sr.RequestError as e:
                raise HTTPException(status_code=500, detail=f"Speech recognition error: {str(e)}")
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
        
        # Process the text through the chat system
        config = {"configurable": {"thread_id": thread_id}}
        
        response_message = None
        for event in graph.stream(
            {"messages": [{"role": "user", "content": text}]}, 
            config, 
            stream_mode="values"
        ):
            if "messages" in event and event["messages"]:
                response_message = event["messages"][-1]
        
        if response_message is None:
            raise HTTPException(status_code=500, detail="No response generated")
        
        # Handle different message types
        response_content = ""
        if hasattr(response_message, 'content'):
            response_content = response_message.content
        elif isinstance(response_message, dict) and 'content' in response_message:
            response_content = response_message['content']
        else:
            response_content = str(response_message)
        
        return ChatResponse(
            response=response_content,
            thread_id=thread_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    
# Alternative text endpoint with proper MongoDB handling
@app.post("/chat/text-with-persistence", response_model=ChatResponse)
async def chat_text_with_persistence(request: TextChatRequest):
    """
    Send a text message to Silica and get a text response (with MongoDB persistence)
    """
    checkpointer = None
    try:
        print("Entering chat/text endpoint with persistence")
        print(f"Request: {request}")
        
        if graph is None:
            raise HTTPException(status_code=500, detail="Graph not initialized")
        
        # Create a fresh checkpointer for this request
        with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
            persistent_graph = create_chat_graph(checkpointer=checkpointer)
        
        # Create a graph with the fresh checkpointer
            config = {"configurable": {"thread_id": request.thread_id}}
            print(f"Config: {config}")
            
            # Create the input message in the correct format
            input_data = {
                "messages": [{"role": "user", "content": request.message}]
            }
            print(f"Input data: {input_data}")
            
            print("Starting graph stream with persistence...")
            response_message = None
            
            # Stream the response from the graph
            try:
                for event in persistent_graph.stream(input_data, config, stream_mode="values"):
                    print(f"Event: {event}")
                    if "messages" in event and event["messages"]:
                        response_message = event["messages"][-1]
                        print(f"Response message: {response_message}")
            except Exception as stream_error:
                print(f"Error during streaming: {stream_error}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Streaming error: {str(stream_error)}")
            
        if response_message is None:
            raise HTTPException(status_code=500, detail="No response generated")
        
        # Handle different message types
        response_content = ""
        if hasattr(response_message, 'content'):
            response_content = response_message.content
        elif isinstance(response_message, dict) and 'content' in response_message:
            response_content = response_message['content']
        else:
            response_content = str(response_message)
        
        return ChatResponse(
            response=response_content,
            thread_id=request.thread_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    finally:
        # Always close the checkpointer connection
        if checkpointer:
            try:
                checkpointer.close()
                print("Checkpointer closed successfully")
            except Exception as close_error:
                print(f"Error closing checkpointer: {close_error}")

# Serve static HTML files
@app.get("/")
async def serve_login():
    """
    Serve the login page
    """
    return FileResponse("login.html")

@app.get("/chat")
async def serve_chat():
    """
    Serve the chat page
    """
    return FileResponse("chat.html")

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "service": "Silica Voice Agent"}

# Get conversation history
@app.get("/chat/history/{thread_id}")
async def get_chat_history(thread_id: str):
    """
    Get conversation history for a specific thread
    """
    try:
        # This would require implementing a method to retrieve conversation history
        # from your MongoDB checkpointer. This is a placeholder implementation.
        return {"thread_id": thread_id, "message": "History retrieval not implemented yet"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)