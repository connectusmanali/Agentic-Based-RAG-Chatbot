import os
from datetime import datetime
from fastapi import FastAPI, Request, Form, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from openai import OpenAI
import tempfile

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

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

# Load vector store and RAG chains
vectorstore = Chroma(
    persist_directory="persisted_rags",
    embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
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

# Fallback-aware RAG response
def get_rag_response(question):
    greetings = ["hi", "hello", "hey", "what's up", "how are you", "good morning", "good evening"]
    if question.strip().lower() in greetings:
        return get_time_based_greeting()

    response = chatbot_chain.invoke({"question": question, "chat_history": []})
    fallback_phrases = [
        "I don't know", "I'm not sure", "I cannot answer that",
        "Sorry, I don't know", "I do not have enough information"
    ]
    if any(phrase.lower() in response['answer'].lower() for phrase in fallback_phrases):
        return "I have no idea about this thing. I am trained on very limited data, that is why I can't answer that question."
    return response['answer']

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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)

    os.remove(tmp_path)
    question = transcript.text
    answer = get_rag_response(question)
    return JSONResponse({"query": question, "answer": answer})