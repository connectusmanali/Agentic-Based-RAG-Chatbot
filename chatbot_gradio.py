import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Load vector store from disk
vectorstore = Chroma(
    persist_directory="persisted_rag",
    embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
)

# Create one persistent chatbot_chain per session
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

# Chat function with custom fallback handling
def chat_with_pdf(message, history):
    response = chatbot_chain.invoke({"question": message, "chat_history": history})

    fallback_phrases = [
        "I don't know",
        "I'm not sure",
        "I cannot answer that",
        "Sorry, I don't know",
        "I do not have enough information"
    ]
    if any(phrase.lower() in response['answer'].lower() for phrase in fallback_phrases):
        return history + [("user", message), ("assistant", "I have no idea about this thing. I am trained on very limited data, that is why I can't answer that question.")]

    return history + [("user", message), ("assistant", response['answer'])]

# Whisper API transcription function
def transcribe_audio(audio_file):
    if audio_file is None:
        return "Please provide an audio input."

    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
    return transcript.text

# Combined function for audio input
def transcribe_and_chat(audio_file, history):
    question = transcribe_audio(audio_file)
    print(f"🗣️ Transcribed Text: {question}")
    return chat_with_pdf(question, history)

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Jovi Realty ChatBot")

    chatbot_display = gr.Chatbot(
        value=[
            ("assistant", "Hey I am Trained on Surrey Zones Data,"),
            ("assistant", "Feel free to ask me anything about Surrey properties Zoning 🏡"),
            ("assistant", "I am still in learning phase")
        ],
        label="Chat History",
        bubble_full_width=False,
        show_label=False,
        avatar_images=(None, None),
        show_copy_button=True
    )

    with gr.Row():
        text_input = gr.Textbox(show_label=False, placeholder="Type your message here...", scale=10)
        send_btn = gr.Button("Send", scale=1)
        mic_btn = gr.Audio(type="filepath", label="", interactive=True, scale=1)

    send_btn.click(
        fn=chat_with_pdf,
        inputs=[text_input, chatbot_display],
        outputs=chatbot_display
    )

    mic_btn.change(
        fn=transcribe_and_chat,
        inputs=[mic_btn, chatbot_display],
        outputs=chatbot_display
    )

# Launch the app
demo.launch(server_name="0.0.0.0", server_port=3000, share=True)