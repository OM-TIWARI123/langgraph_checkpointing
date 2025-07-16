import asyncio
import json
import base64
import websockets
import os
import PyPDF2
import docx
from typing import Annotated, AsyncGenerator, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import threading
import queue
import pygame
from io import BytesIO
import re
import time
from datetime import datetime
import chromadb

load_dotenv()

class InterviewState(TypedDict):
    messages: Annotated[list, add_messages]
    resume_content: str
    resume_chunks: List[Dict]
    role: str
    current_question_index: int
    questions_queue: List[Dict]
    responses: List[Dict]
    interview_phase: str  # "resume_processing", "introduction", "questioning", "feedback", "analytics"
    session_metadata: Dict
    performance_scores: Dict
    feedback_data: Dict
    vector_store: Any
    resume_path: str

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
voice_id = 'Xb7hH8MSUJpSbSDYk0k2'
model_id = 'eleven_flash_v2_5'

llm = init_chat_model("google_genai:gemini-2.0-flash")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Role-specific question templates
ROLE_TEMPLATES = {
    "SDE": {
        "technical": [
            "data structures", "algorithms", "system design", "coding practices",
            "debugging", "performance optimization", "testing", "version control"
        ],
        "behavioral": [
            "problem solving", "teamwork", "code reviews", "project management",
            "learning new technologies", "handling deadlines"
        ],
        "experience_based": [
            "challenging projects", "technical decisions", "code architecture",
            "debugging complex issues", "performance improvements"
        ]
    },
    "Data Scientist": {
        "technical": [
            "machine learning", "statistics", "data preprocessing", "model evaluation",
            "feature engineering", "deep learning", "data visualization", "SQL"
        ],
        "behavioral": [
            "data storytelling", "stakeholder communication", "project prioritization",
            "model deployment", "handling ambiguous problems"
        ],
        "experience_based": [
            "data projects", "model performance", "business impact", "data pipeline",
            "A/B testing", "model optimization"
        ]
    },
    "Product Manager": {
        "technical": [
            "product strategy", "user research", "data analysis", "prioritization",
            "roadmap planning", "metrics definition", "A/B testing"
        ],
        "behavioral": [
            "stakeholder management", "cross-functional collaboration", "decision making",
            "conflict resolution", "user empathy", "strategic thinking"
        ],
        "experience_based": [
            "product launches", "feature prioritization", "user feedback",
            "market analysis", "product metrics", "team leadership"
        ]
    }
}

class AudioStreamer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.audio_thread = None
        
    def start_audio_thread(self):
        if not self.is_playing:
            self.is_playing = True
            self.audio_thread = threading.Thread(target=self._audio_player)
            self.audio_thread.daemon = True
            self.audio_thread.start()
    
    def _audio_player(self):
        while self.is_playing:
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                if audio_chunk is None:
                    break
                
                audio_io = BytesIO(audio_chunk)
                pygame.mixer.music.load(audio_io)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio playback error: {e}")
    
    def add_audio_chunk(self, chunk):
        self.audio_queue.put(chunk)
    
    def stop(self):
        self.is_playing = False
        self.audio_queue.put(None)

audio_streamer = AudioStreamer()

async def text_to_speech_websocket(text_stream: AsyncGenerator[str, None]):
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}"
    
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({
                "text": " ",
                "voice_settings": {
                    "stability": 0.5, 
                    "similarity_boost": 0.8, 
                    "use_speaker_boost": False
                },
                "generation_config": {
                    "chunk_length_schedule": [50, 120, 160, 290]
                },
                "xi_api_key": ELEVENLABS_API_KEY,
            }))
            
            audio_streamer.start_audio_thread()
            
            send_task = asyncio.create_task(send_text_stream(websocket, text_stream))
            receive_task = asyncio.create_task(receive_audio_stream(websocket))
            
            await asyncio.gather(send_task, receive_task)
            
    except Exception as e:
        print(f"WebSocket error: {e}")

async def send_text_stream(websocket, text_stream: AsyncGenerator[str, None]):
    try:
        async for text_chunk in text_stream:
            if text_chunk:
                await websocket.send(json.dumps({"text": text_chunk}))
        
        await websocket.send(json.dumps({"text": "", "flush": True}))
        
    except Exception as e:
        print(f"Error sending text: {e}")

async def receive_audio_stream(websocket):
    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data.get("audio"):
                audio_chunk = base64.b64decode(data["audio"])
                audio_streamer.add_audio_chunk(audio_chunk)
                
            elif data.get('isFinal'):
                break
                
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"Error receiving audio: {e}")

async def async_text_generator(text_chunks):
    for chunk in text_chunks:
        yield chunk
        await asyncio.sleep(0.01)

def generate_audio_for_text(text):
    """Generate audio for given text"""
    def run_audio_generation():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Split text into chunks for better streaming
            chunks = [text[i:i+50] for i in range(0, len(text), 50)]
            loop.run_until_complete(
                text_to_speech_websocket(async_text_generator(chunks))
            )
        finally:
            loop.close()
    
    audio_thread = threading.Thread(target=run_audio_generation)
    audio_thread.daemon = True
    audio_thread.start()

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return ""

def resume_processor(state: InterviewState):
    """Process resume and store in vector database"""
    print("📄 Processing resume...")
    
    resume_path = state.get("resume_path", "")
    if not resume_path or not os.path.exists(resume_path):
        return {
            "interview_phase": "error",
            "messages": [AIMessage(content="Resume file not found. Please provide a valid file path.")]
        }
    
    # Extract text based on file type
    if resume_path.lower().endswith('.pdf'):
        resume_text = extract_text_from_pdf(resume_path)
    elif resume_path.lower().endswith('.docx'):
        resume_text = extract_text_from_docx(resume_path)
    else:
        with open(resume_path, 'r', encoding='utf-8') as file:
            resume_text = file.read()
    
    # Chunk the resume
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_text(resume_text)
    
    # Create vector store
    try:
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=f"resume_{int(time.time())}"
        )
        
        # Extract key information
        key_info = extract_key_resume_info(resume_text)
        
        print("✅ Resume processed successfully!")
        
        return {
            "resume_content": resume_text,
            "resume_chunks": [{"text": chunk, "index": i} for i, chunk in enumerate(chunks)],
            "vector_store": vector_store,
            "session_metadata": {
                "start_time": datetime.now().isoformat(),
                "resume_length": len(resume_text),
                "num_chunks": len(chunks),
                "key_info": key_info
            },
            "interview_phase": "introduction",
            "current_question_index": 0,
            "questions_queue": [],
            "responses": [],
            "performance_scores": {},
            "feedback_data": {}
        }
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {
            "interview_phase": "error",
            "messages": [AIMessage(content=f"Error processing resume: {str(e)}")]
        }

def extract_key_resume_info(resume_text):
    """Extract key information from resume"""
    info = {
        "experience_years": 0,
        "key_skills": [],
        "education": [],
        "companies": []
    }
    
    # Simple regex patterns (can be enhanced)
    experience_pattern = r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)'
    skills_pattern = r'(?:skills?|technologies?|tools?)[:\s]*([^\n]+)'
    education_pattern = r'(?:education|degree|bachelor|master|phd)[:\s]*([^\n]+)'
    
    # Extract years of experience
    exp_matches = re.findall(experience_pattern, resume_text, re.IGNORECASE)
    if exp_matches:
        info["experience_years"] = max([int(x) for x in exp_matches])
    
    # Extract skills
    skills_matches = re.findall(skills_pattern, resume_text, re.IGNORECASE)
    for match in skills_matches:
        skills = [skill.strip() for skill in match.split(',')]
        info["key_skills"].extend(skills)
    
    return info

def introduction(state: InterviewState):
    """Generate personalized introduction"""
    print("👋 Generating introduction...")
    
    role = state.get("role", "Software Engineer")
    key_info = state.get("session_metadata", {}).get("key_info", {})
    
    system_prompt = f"""
    You are an experienced interviewer conducting a {role} interview. 
    Generate a warm, professional introduction that:
    1. Welcomes the candidate
    2. Explains this is a resume-based interview
    3. Mentions you'll be asking questions about their experience and skills
    4. Sets expectations for the interview duration (15-20 minutes)
    5. Encourages them to be detailed in their responses
    
    Keep it conversational and encouraging. Make it about 2-3 sentences.
    """
    
    stream = llm.stream([SystemMessage(content=system_prompt)])
    full_content = ""
    text_chunks = []
    
    for chunk in stream:
        content_piece = chunk.content if hasattr(chunk, "content") else str(chunk)
        if content_piece:
            full_content += content_piece
            text_chunks.append(content_piece)
            print(content_piece, end="", flush=True)
    
    print()
    
    # Generate audio
    if text_chunks:
        generate_audio_for_text(full_content)
    
    return {
        "messages": [AIMessage(content=full_content)],
        "interview_phase": "questioning"
    }

def question_generator(state: InterviewState):
    """Generate questions based on resume and role"""
    print("🤔 Generating interview questions...")
    
    role = state.get("role", "SDE")
    vector_store = state.get("vector_store")
    resume_content = state.get("resume_content", "")
    
    if not vector_store:
        return {"interview_phase": "error"}
    
    # Get role-specific topics
    role_topics = ROLE_TEMPLATES.get(role, ROLE_TEMPLATES["SDE"])
    
    # Generate questions for each category
    questions = []
    
    # Technical questions based on resume
    technical_questions = generate_technical_questions(resume_content, role, vector_store)
    questions.extend(technical_questions)
    
    # Experience-based questions
    experience_questions = generate_experience_questions(resume_content, role, vector_store)
    questions.extend(experience_questions)
    
    # Behavioral questions
    behavioral_questions = generate_behavioral_questions(role)
    questions.extend(behavioral_questions)
    
    print(f"✅ Generated {len(questions)} questions")
    
    return {
        "questions_queue": questions,
        "interview_phase": "questioning"
    }

def generate_technical_questions(resume_content, role, vector_store):
    """Generate technical questions based on resume content"""
    questions = []
    
    # Search for technical content in resume
    technical_topics = ROLE_TEMPLATES.get(role, {}).get("technical", [])
    
    for topic in technical_topics[:3]:  # Limit to 3 technical questions
        docs = vector_store.similarity_search(topic, k=2)
        if docs:
            context = docs[0].page_content
            question = f"I see you have experience with {topic}. Can you walk me through a specific project or situation where you used {topic}? What challenges did you face and how did you overcome them?"
            questions.append({
                "question": question,
                "category": "technical",
                "topic": topic,
                "context": context,
                "difficulty": "medium"
            })
    
    return questions

def generate_experience_questions(resume_content, role, vector_store):
    """Generate experience-based questions"""
    questions = []
    
    # Search for project/experience content
    experience_queries = ["project", "experience", "work", "developed", "implemented"]
    
    for query in experience_queries[:2]:  # Limit to 2 experience questions
        docs = vector_store.similarity_search(query, k=1)
        if docs:
            context = docs[0].page_content
            question = f"Based on your resume, I'd like to know more about your experience with {query}. Can you elaborate on a specific example and what your role was?"
            questions.append({
                "question": question,
                "category": "experience",
                "topic": query,
                "context": context,
                "difficulty": "medium"
            })
    
    return questions

def generate_behavioral_questions(role):
    """Generate behavioral questions"""
    behavioral_questions = [
        "Tell me about a time when you had to work under a tight deadline. How did you handle the pressure?",
        "Describe a situation where you had to learn a new technology or skill quickly. What was your approach?",
        "Can you share an example of when you had to collaborate with a difficult team member? How did you handle it?",
        "Tell me about a project that didn't go as planned. What did you learn from it?"
    ]
    
    questions = []
    for q in behavioral_questions[:2]:  # Limit to 2 behavioral questions
        questions.append({
            "question": q,
            "category": "behavioral",
            "topic": "soft_skills",
            "context": "",
            "difficulty": "medium"
        })
    
    return questions

def interview_conductor(state: InterviewState):
    """Conduct the interview by asking questions"""
    questions_queue = state.get("questions_queue", [])
    current_index = state.get("current_question_index", 0)
    
    if current_index >= len(questions_queue):
        return {"interview_phase": "feedback"}
    
    current_question = questions_queue[current_index]
    question_text = current_question["question"]
    
    print(f"\n🎯 Question {current_index + 1}/{len(questions_queue)}")
    print(f"Category: {current_question['category']}")
    print(f"❓ {question_text}")
    
    # Generate audio for question
    generate_audio_for_text(question_text)
    
    return {
        "messages": [AIMessage(content=question_text)],
        "interview_phase": "waiting_for_response"
    }

def process_response(state: InterviewState, user_response: str):
    """Process user response and evaluate it"""
    questions_queue = state.get("questions_queue", [])
    current_index = state.get("current_question_index", 0)
    responses = state.get("responses", [])
    
    if current_index >= len(questions_queue):
        return {"interview_phase": "feedback"}
    
    current_question = questions_queue[current_index]
    
    # Evaluate the response
    evaluation = evaluate_response(user_response, current_question, state)
    
    # Store the response
    response_data = {
        "question": current_question,
        "answer": user_response,
        "evaluation": evaluation,
        "timestamp": datetime.now().isoformat()
    }
    
    responses.append(response_data)
    
    # Move to next question
    next_index = current_index + 1
    
    if next_index >= len(questions_queue):
        return {
            "responses": responses,
            "current_question_index": next_index,
            "interview_phase": "feedback"
        }
    else:
        return {
            "responses": responses,
            "current_question_index": next_index,
            "interview_phase": "questioning"
        }

def evaluate_response(response: str, question: Dict, state: InterviewState):
    """Evaluate user response using LLM"""
    evaluation_prompt = f"""
    Evaluate this interview response on a scale of 1-10 for the following criteria:
    1. Relevance to the question
    2. Technical accuracy (if applicable)
    3. Communication clarity
    4. Depth of knowledge
    5. Problem-solving approach
    
    Question: {question['question']}
    Category: {question['category']}
    Response: {response}
    
    Provide scores for each criterion and brief feedback. Format as JSON:
    {{
        "relevance": score,
        "technical_accuracy": score,
        "communication": score,
        "depth": score,
        "problem_solving": score,
        "overall_score": average_score,
        "feedback": "detailed feedback here",
        "strengths": ["strength1", "strength2"],
        "improvements": ["improvement1", "improvement2"]
    }}
    """
    
    try:
        result = llm.invoke([SystemMessage(content=evaluation_prompt)])
        evaluation = json.loads(result.content)
        return evaluation
    except Exception as e:
        print(f"Error evaluating response: {e}")
        return {
            "relevance": 5,
            "technical_accuracy": 5,
            "communication": 5,
            "depth": 5,
            "problem_solving": 5,
            "overall_score": 5,
            "feedback": "Could not evaluate response automatically.",
            "strengths": [],
            "improvements": []
        }

def feedback_analyzer(state: InterviewState):
    """Analyze all responses and provide feedback"""
    print("📊 Analyzing interview performance...")
    
    responses = state.get("responses", [])
    if not responses:
        return {"interview_phase": "analytics"}
    
    # Calculate overall performance
    total_score = 0
    category_scores = {}
    
    for response in responses:
        evaluation = response.get("evaluation", {})
        score = evaluation.get("overall_score", 0)
        total_score += score
        
        category = response["question"]["category"]
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(score)
    
    # Calculate averages
    overall_average = total_score / len(responses) if responses else 0
    category_averages = {cat: sum(scores)/len(scores) for cat, scores in category_scores.items()}
    
    # Generate feedback summary
    feedback_prompt = f"""
    Based on the interview performance, provide a comprehensive feedback summary:
    
    Overall Score: {overall_average:.1f}/10
    Category Scores: {category_averages}
    
    Number of questions answered: {len(responses)}
    
    Provide:
    1. Overall performance summary
    2. Key strengths demonstrated
    3. Areas for improvement
    4. Specific recommendations
    5. Next steps for the candidate
    
    Make it constructive and encouraging while being honest about areas needing work.
    """
    
    stream = llm.stream([SystemMessage(content=feedback_prompt)])
    full_content = ""
    text_chunks = []
    
    for chunk in stream:
        content_piece = chunk.content if hasattr(chunk, "content") else str(chunk)
        if content_piece:
            full_content += content_piece
            text_chunks.append(content_piece)
            print(content_piece, end="", flush=True)
    
    print()
    
    # Generate audio
    if text_chunks:
        generate_audio_for_text(full_content)
    
    return {
        "messages": [AIMessage(content=full_content)],
        "performance_scores": {
            "overall_average": overall_average,
            "category_averages": category_averages,
            "total_questions": len(responses)
        },
        "feedback_data": {
            "summary": full_content,
            "detailed_responses": responses
        },
        "interview_phase": "analytics"
    }

def analytics_generator(state: InterviewState):
    """Generate final analytics and report"""
    print("📈 Generating analytics report...")
    
    performance_scores = state.get("performance_scores", {})
    feedback_data = state.get("feedback_data", {})
    session_metadata = state.get("session_metadata", {})
    
    # Create detailed report
    report = f"""
    
    =================== INTERVIEW ANALYTICS REPORT ===================
    
    📊 PERFORMANCE SUMMARY:
    • Overall Score: {performance_scores.get('overall_average', 0):.1f}/10
    • Questions Answered: {performance_scores.get('total_questions', 0)}
    • Interview Duration: {session_metadata.get('start_time', 'N/A')}
    
    📈 CATEGORY BREAKDOWN:
    """
    
    for category, score in performance_scores.get('category_averages', {}).items():
        report += f"    • {category.title()}: {score:.1f}/10\n"
    
    report += f"""
    
    💡 DETAILED FEEDBACK:
    {feedback_data.get('summary', 'No feedback available')}
    
    📋 RECOMMENDATIONS:
    • Practice more technical questions in weak areas
    • Work on communication clarity
    • Prepare more detailed examples from your experience
    • Review fundamental concepts for your role
    
    =================== END OF REPORT ===================
    """
    
    print(report)
    
    # Generate audio for key insights
    key_insights = f"Your interview is complete! Your overall score is {performance_scores.get('overall_average', 0):.1f} out of 10. Check the detailed report for specific feedback and recommendations."
    generate_audio_for_text(key_insights)
    
    return {
        "messages": [AIMessage(content=report)],
        "interview_phase": "completed"
    }

# Build the graph
graph_builder = StateGraph(InterviewState)

# Add nodes
graph_builder.add_node("resume_processor", resume_processor)
graph_builder.add_node("introduction", introduction)
graph_builder.add_node("question_generator", question_generator)
graph_builder.add_node("interview_conductor", interview_conductor)
graph_builder.add_node("feedback_analyzer", feedback_analyzer)
graph_builder.add_node("analytics_generator", analytics_generator)

# Add edges
graph_builder.add_edge(START, "resume_processor")
graph_builder.add_edge("resume_processor", "introduction")
graph_builder.add_edge("introduction", "question_generator")
graph_builder.add_edge("question_generator", "interview_conductor")
graph_builder.add_edge("feedback_analyzer", "analytics_generator")
graph_builder.add_edge("analytics_generator", END)

# Conditional edge for interview loop
def should_continue_interview(state):
    phase = state.get("interview_phase", "")
    if phase == "questioning":
        return "interview_conductor"
    elif phase == "waiting_for_response":
        return "waiting_for_response"
    elif phase == "feedback":
        return "feedback_analyzer"
    else:
        return "end"

graph_builder.add_conditional_edges(
    "interview_conductor",
    should_continue_interview,
    {
        "interview_conductor": "interview_conductor",
        "waiting_for_response": "interview_conductor",
        "feedback_analyzer": "feedback_analyzer",
        "end": END
    }
)

def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)