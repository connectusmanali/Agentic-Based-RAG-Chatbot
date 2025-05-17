import os
import logging
from datetime import datetime
from fastapi import FastAPI, Request, Form, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# Verify environment variables
required_env_vars = {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME
}
for var_name, var_value in required_env_vars.items():
    if not var_value:
        logger.error(f"Missing required environment variable: {var_name}")
        raise ValueError(f"Missing required environment variable: {var_name}")

# Initialize OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

# Initialize Pinecone
try:
    pc = Pinecone(api_key=required_env_vars["PINECONE_API_KEY"])
    logger.info("Pinecone initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {str(e)}")
    raise

# Init FastAPI and templates
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vector store and RAG chain
try:
    embedding = OpenAIEmbeddings(
        api_key=required_env_vars["OPENAI_API_KEY"],
        model="text-embedding-3-small"
    )
    vectorstore = PineconeVectorStore(
        index_name=required_env_vars["PINECONE_INDEX_NAME"],
        embedding=embedding,
        namespace=PINECONE_NAMESPACE,
        pinecone_api_key=required_env_vars["PINECONE_API_KEY"]
    )
    logger.info("Vector store initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {str(e)}")
    raise

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
try:
    chatbot_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            model_name="gpt-4o-mini",
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=300
        ),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory
    )
    logger.info("ConversationalRetrievalChain initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ConversationalRetrievalChain: {str(e)}")
    raise

# Fallback-aware RAG response
def get_rag_response(question):
    greetings = ["hi", "hello", "hey", "what's up", "how are you", "good morning", "good evening"]
    if question.strip().lower() in greetings:
        return get_time_based_greeting()

    try:
        response = chatbot_chain.invoke({"question": question, "chat_history": []})
        fallback_phrases = [
            "I don't know", "I'm not sure", "I cannot answer that",
            "Sorry, I don't know", "I do not have enough information"
        ]
        if any(phrase.lower() in response['answer'].lower() for phrase in fallback_phrases):
            return "I have no idea about this thing. I am trained on very limited data, that is why I can't answer that question."
        return response['answer']
    except Exception as e:
        logger.error(f"Error in get_rag_response: {str(e)}")
        return "An error occurred while processing your question."

# Time-based greeting message
def get_time_based_greeting():
    hour = datetime.now().hour
    if hour < 12:
        greeting = "🌅 Good morning"
    elif 12 <= hour < 18:
        greeting = "🌞 Good afternoon"
    else:
        greeting = "🌇 Good evening"
    return greeting

# Initial welcome messages
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request, ended: bool = Query(False)):
    if ended:
        return templates.TemplateResponse("chat_end.html", {"request": request})
    time_greeting = get_time_based_greeting()
    initial_messages = [
        f"{time_greeting}! I'm **JoviBot**.",
        "💬 You can ask me anything about our products, get support, and track your order."
    ]
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "initial_messages": initial_messages
    })

@app.post("/api/query")
async def query_api(request: Request):
    form_data = await request.form()
    message = form_data.get("message")
    if not message:
        return JSONResponse(status_code=400, content={"error": "Message is required"})
    answer = get_rag_response(message)
    return JSONResponse({"answer": answer})

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=f)

        os.remove(tmp_path)
        question = transcript.text
        answer = get_rag_response(question)
        return JSONResponse({"query": question, "answer": answer})
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Failed to process audio"})