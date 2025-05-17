---
title: ChatBot
app_file: chatbot_gradio.py
sdk: gradio
sdk_version: 5.25.2
---

# 🧠 RAG Chatbot with FastAPI & React

This project is an AI-powered chatbot built using **Retrieval-Augmented Generation (RAG)**, a **Vector Database**, **FastAPI** for the backend, and **React** for the frontend. It enables users to ask natural language questions and get accurate, context-aware responses from uploaded documents or pre-loaded knowledge sources.

---

## 🚀 Features

- Upload and process documents (PDF, TXT, etc.)
- Generate embeddings and store them in a vector database
- Retrieve relevant chunks via semantic search
- Answer queries using LLMs with retrieved context
- Interactive React-based chatbot UI
- Scalable and modular architecture

---

## 🧱 Tech Stack

- **Frontend**: React (Vite or CRA)
- **Backend**: FastAPI
- **Vector Database**: FAISS / Pinecone / Chroma
- **Embeddings**: OpenAI / Hugging Face / Sentence Transformers
- **LLM**: OpenAI GPT / Other generative models

---

## 📁 Project Structure

```plaintext
rag-chatbot/
│
├── backend/
│   ├── main.py            # FastAPI app entrypoint
│   ├── rag.py             # RAG core logic (retrieval + generation)
│   ├── db/                # Vector DB utilities
│   └── utils/             # Text chunking, embedding, etc.
│
├── frontend/
│   ├── src/
│   │   ├── components/    # Chat interface
│   │   ├── App.jsx
│   │   └── index.jsx
│
└── README.md

⚙️ Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
2. Backend Setup (FastAPI)
bash
Copy
Edit
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Create a .env file:

env
Copy
Edit
OPENAI_API_KEY=your_openai_api_key
VECTOR_DB_PATH=./data/faiss_index
Run the FastAPI server:

bash
Copy
Edit
uvicorn main:app --reload
3. Frontend Setup (React)
bash
Copy
Edit
cd frontend
npm install
Create a .env file in the frontend directory:

env
Copy
Edit
VITE_API_URL=http://localhost:8000
Start the React development server:

bash
Copy
Edit
npm run dev
✅ To-Do
 Add authentication

 Support multi-file upload

 Enable LLM streaming responses

 Store chat history

 Add support for DOCX, HTML files

🤝 Contributing
Contributions are welcome! Feel free to fork the repo, open issues, or submit pull requests.
