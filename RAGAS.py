import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
import json

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load vector store
vectorstore = Chroma(
    persist_directory="persisted_rag",
    embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
)
# Setup memory and chatbot chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    max_tokens=300
)
chatbot_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)
# Evaluation log
eval_log = []
def chat_with_pdf_and_log(message, history):
    result = chatbot_chain({"question": message})
    answer = result["answer"]
    contexts = [doc.page_content for doc in result["source_documents"]]
    # Add to evaluation log
    eval_log.append({
        "question": message,
        "answer": answer,
        "contexts": contexts
    })
    fallback_phrases = [
        "I don't know", 
        "I'm not sure", 
        "I cannot answer that",
        "Sorry, I don't know", 
        "I do not have enough information"
    ]
    if any(phrase.lower() in answer.lower() for phrase in fallback_phrases):
        return "I have no idea about this thing. I am trained on very limited data, that is why I can't answer that question."
    return answer
# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Jovi Realty ChatBot with RAGAS Evaluation")
    chatbot = gr.ChatInterface(fn=chat_with_pdf_and_log)
    gr.Button("Evaluate with RAGAS").click(
        lambda: evaluate(
            Dataset.from_list(eval_log),
            metrics=[faithfulness, answer_relevancy, context_precision]
        ).to_pandas().to_json(orient="records", indent=2),
        outputs=gr.Textbox(label="Evaluation Results (JSON)", lines=10)
    )
# Launch app
demo.launch(server_name="0.0.0.0", server_port=8000, share=True)