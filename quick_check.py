import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Verify environment variables
required_env_vars = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME")
}
for var_name, var_value in required_env_vars.items():
    if not var_value:
        raise ValueError(f"Missing required environment variable: {var_name}")

PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# Initialize Pinecone
try:
    pc = Pinecone(
        api_key=required_env_vars["PINECONE_API_KEY"]
    )
except Exception as e:
    raise Exception(f"Failed to initialize Pinecone: {str(e)}")

# Initialize embedding and vector store
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
except Exception as e:
    raise Exception(f"Failed to initialize vector store: {str(e)}")

# Create retriever and QA chain
try:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        api_key=required_env_vars["OPENAI_API_KEY"],
        model="gpt-4o-mini",
        temperature=0.7
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
except Exception as e:
    raise Exception(f"Failed to initialize QA chain: {str(e)}")

# Ask a question
def main():
    try:
        query = input("❓ Ask a question: ")
        if not query.strip():
            print("Please enter a valid question.")
            return
            
        result = qa.invoke({"query": query})
        answer = result["result"]
        sources = result.get("source_documents", [])
        
        print("\n🧠 Answer:")
        print(answer)
        
        if sources:
            print("\n📚 Sources:")
            for i, doc in enumerate(sources, 1):
                print(f"{i}. {doc.metadata.get('source', 'Unknown source')}")
                
    except Exception as e:
        print(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    main()