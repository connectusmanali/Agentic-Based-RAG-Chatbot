import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

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

# Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
# Load and split PDFs
pdf_folder = "data"
all_documents = []

try:
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            logger.info(f"📄 Processing: {filename}")
            loader = PyPDFLoader(path)
            docs = loader.load_and_split(text_splitter=text_splitter)
            for doc in docs:
                doc.metadata["source"] = filename
            all_documents.extend(docs)
except Exception as e:
    logger.error(f"Error processing PDFs: {str(e)}")
    raise

logger.info(f"🧠 Total chunks: {len(all_documents)}")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Pinecone initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {str(e)}")
    raise

# Create index if it doesn't exist
try:
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        logger.info(f"🆕 Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust region if needed
        )
except Exception as e:
    logger.error(f"Failed to create Pinecone index: {str(e)}")
    raise

# Push to Pinecone using LangChain in batches
try:
    logger.info("📤 Uploading embeddings to Pinecone...")
    embedding = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )
    
    # Batch documents to avoid payload size limit
    batch_size = 100
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i + batch_size]
        logger.info(f"Uploading batch {i//batch_size + 1} of {len(all_documents)//batch_size + 1} ({len(batch)} documents)")
        vectorstore = PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embedding,
            index_name=PINECONE_INDEX_NAME,
            namespace=PINECONE_NAMESPACE
        )
    logger.info("✅ Vector store created and uploaded to Pinecone.")
except Exception as e:
    logger.error(f"Failed to upload embeddings: {str(e)}")
    raise