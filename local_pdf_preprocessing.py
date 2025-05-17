# preprocess_pdf.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Set up better text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

# 2. Load multiple PDFs
pdf_folder = "data"
all_documents = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(pdf_folder, filename)
        print(f"📄 Processing: {filename}")
        loader = PyPDFLoader(path)
        docs = loader.load_and_split(text_splitter=text_splitter)
        for doc in docs:
            doc.metadata["source"] = filename
        all_documents.extend(docs)

# 3. Build and save vectorstore
print(f"🧠 Total chunks: {len(all_documents)}")
vectorstore = Chroma.from_documents(
    all_documents,
    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
    persist_directory="persisted_rags"
)
print("✅ Vector store created and saved.")