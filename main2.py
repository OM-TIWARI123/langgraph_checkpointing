from dotenv import load_dotenv
import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
from graph2 import create_chat_graph, audio_streamer, process_response
from google import genai
import os
import time
import atexit

load_dotenv()

# Fix the API key issue
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

MONGODB_URI = "mongodb://admin:admin@localhost:27017"
config = {"configurable": {"thread_id": "interview_session"}}

def cleanup():
    """Cleanup function to stop audio when exiting"""
    audio_streamer.stop()

# Register cleanup function
atexit.register(cleanup)

def get_user_input():
    """Get resume path and role from user"""
    print("=== AI Interview System ===")
    print("Welcome to the AI-powered interview system!")
    print()
    
    # Get resume file path
    while True:
        resume_path = input("📄 Enter the path to your resume (PDF/DOCX/TXT): ").strip()
        if os.path.exists(resume_path):
            break
        else:
            print("❌ File not found. Please enter a valid file path.")
    
    # Get role
    print("\n📋 Available roles:")
    roles = ["SDE", "Data Scientist", "Product Manager"]
    for i, role in enumerate(roles, 1):
        print(f"  {i}. {role}")
    
    while True:
        try:
            choice = input("\n🎯 Select role (1-3): ").strip()
            role_index = int(choice) - 1
            if 0 <= role_index < len(roles):
                selected_role = roles[role_index]
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    return resume_path, selected_role

def listen_for_speech(r, microphone, timeout=30):
    """Listen for speech input with timeout"""
    print("🎤 Listening... (speak now)")
    print("💡 Press Ctrl+C to skip to next question")
    
    try:
        # Listen for audio input
        audio = r.listen(microphone, timeout=timeout, phrase_time_limit=60)
        print("🔄 Processing speech...")
        
        # Recognize speech using Google
        text = r.recognize_google(audio)
        print(f"📝 You said: {text}")
        return text
        
    except sr.UnknownValueError:
        print("❌ Could not understand audio. Please try again.")
        return None
    except sr.RequestError as e:
        print(f"❌ Could not request results from Google Speech Recognition service; {e}")
        return None
    except sr.WaitTimeoutError:
        print("⏰ Listening timeout. Moving to next question...")
        return None
    except KeyboardInterrupt:
        print("\n⏭️ Skipping question...")
        return None

def handle_interview_interaction(graph, current_state, r, microphone, config):
    """Handle the interactive interview process"""
    
    while True:
        phase = current_state.get("interview_phase", "")
        
        if phase == "questioning":
            # Ask the current question
            for event in graph.stream(current_state, config, stream_mode="values"):
                current_state = event
                if current_state.get("interview_phase") == "waiting_for_response":
                    break
            
            # Wait for user response
            print("\n" + "="*50)
            response = listen_for_speech(r, microphone)
            
            if response:
                print(f"✅ Response recorded: {response[:100]}...")
                
                # Process the response
                updated_state = process_response(current_state, response)
                current_state.update(updated_state)
                
                # Small delay before next question
                time.sleep(2)
                
                # Check if we should continue or move to feedback
                if current_state.get("interview_phase") == "feedback":
                    break
                    
            else:
                print("⏭️ No response recorded. Moving to next question...")
                # Move to next question anyway
                questions_queue = current_state.get("questions_queue", [])
                current_index = current_state.get("current_question_index", 0)
                
                if current_index + 1 >= len(questions_queue):
                    current_state["interview_phase"] = "feedback"
                    break
                else:
                    current_state["current_question_index"] = current_index + 1
                    current_state["interview_phase"] = "questioning"
        
        elif phase == "waiting_for_response":
            # This should be handled above
            continue
            
        elif phase == "feedback":
            # Move to feedback phase
            break
            
        else:
            # Unexpected phase
            print(f"⚠️ Unexpected interview phase: {phase}")
            break
    
    return current_state

def main():
    print("🚀 Starting AI Interview System...")
    print("Make sure your microphone is working and MongoDB is running.")
    print()
    
    # Get user inputs
    resume_path, role = get_user_input()
    
    print(f"\n✅ Resume: {resume_path}")
    print(f"✅ Role: {role}")
    print("\n🔄 Initializing interview system...")
    
    try:
        with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
            graph = create_chat_graph(checkpointer=checkpointer)
            
            # Initialize speech recognition
            r = sr.Recognizer()
            r.energy_threshold = 300
            r.dynamic_energy_threshold = True
            r.pause_threshold = 1.5
            r.operation_timeout = None
            r.phrase_threshold = 0.3
            r.non_speaking_duration = 0.8
            
            # Initialize interview state
            initial_state = {
                "messages": [],
                "resume_path": resume_path,
                "role": role,
                "interview_phase": "resume_processing",
                "current_question_index": 0,
                "questions_queue": [],
                "responses": [],
                "performance_scores": {},
                "feedback_data": {},
                "session_metadata": {}
            }
            
            print("🔄 Processing resume and starting interview...")
            
            # Start the interview process
            current_state = initial_state
            
            with sr.Microphone() as source:
                print("🎤 Calibrating microphone...")
                r.adjust_for_ambient_noise(source, duration=2)
                print("✅ Microphone calibrated!")
                
                # Process initial phases (resume processing, introduction, question generation)
                for event in graph.stream(current_state, config, stream_mode="values"):
                    current_state = event
                    phase = current_state.get("interview_phase", "")
                    
                    print(f"📍 Current phase: {phase}")
                    
                    # Stop at questioning phase to start interaction
                    if phase == "questioning":
                        print("🎯 Starting interview questions...")
                        break
                    elif phase == "error":
                        print("❌ Error occurred during setup")
                        return
                
                # Handle interactive interview
                if current_state.get("interview_phase") == "questioning":
                    print("\n🎙️ Interview will now begin!")
                    print("📝 Instructions:")
                    print("   • Speak clearly into your microphone")
                    print("   • You have 30 seconds per question")
                    print("   • Press Ctrl+C to skip a question")
                    print("   • Be detailed in your responses")
                    print("\n" + "="*50)
                    
                    # Start the interactive interview process
                    current_state = handle_interview_interaction(graph, current_state, r, source, config)
                
                # Process feedback and analytics phases
                print("\n🔄 Generating feedback and analytics...")
                for event in graph.stream(current_state, config, stream_mode="values"):
                    current_state = event
                    phase = current_state.get("interview_phase", "")
                    
                    if phase == "completed":
                        print("✅ Interview completed successfully!")
                        break
                        
                # Wait for audio to finish
                print("\n🔊 Waiting for audio to complete...")
                time.sleep(3)
                
                # Final summary
                performance_scores = current_state.get("performance_scores", {})
                total_questions = performance_scores.get("total_questions", 0)
                overall_score = performance_scores.get("overall_average", 0)
                
                print("\n" + "="*60)
                print("🎉 INTERVIEW COMPLETED!")
                print(f"📊 Questions Answered: {total_questions}")
                print(f"🎯 Overall Score: {overall_score:.1f}/10")
                print("📋 Check the detailed report above for comprehensive feedback.")
                print("="*60)
                
    except KeyboardInterrupt:
        print("\n\n⏹️ Interview interrupted by user.")
        print("Thank you for using the AI Interview System!")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your configuration and try again.")
        
    finally:
        # Cleanup
        cleanup()
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()