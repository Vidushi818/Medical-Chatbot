from flask import Flask, render_template, request
from langchain_ollama import ChatOllama
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from collections import defaultdict
from src.prompt import *
import os


# -------------------- APP INIT --------------------
app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# -------------------- LLM + RAG SETUP --------------------
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# smaller context → shorter answers
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

chatModel = ChatOllama(
    model="tinyllama",
    temperature=0.1
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# -------------------- MEMORY --------------------
chat_memory = defaultdict(list)

def build_memory_context(session_id):
    history = chat_memory[session_id][-4:]  # last 4 exchanges
    context = ""

    for role, text in history:
        context += f"{role}: {text}\n"

    return context


# -------------------- SYMPTOM DETECTOR --------------------
def is_symptom_query(text: str):
    symptoms = [
        "pain", "fever", "headache", "vomiting", "nausea",
        "cough", "cold", "dizziness", "weakness", "infection",
        "hurt", "burning", "swelling", "rash", "bleeding",
        "stomach", "throat", "body ache"
    ]
    text = text.lower()
    return any(word in text for word in symptoms)


# -------------------- INTENT CLASSIFIER --------------------
def classify_query(user_input: str):
    text = user_input.lower().strip()

    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    identity = ["who are you", "your name", "what are you", "what can you do"]

    if any(g in text for g in greetings):
        return "greeting"

    if any(i in text for i in identity):
        return "identity"

    if is_symptom_query(text):
        return "symptom"

    if len(text.split()) <= 2:
        return "chat"

    return "medical"


# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    session_id = request.remote_addr

    intent = classify_query(user_input)
    print("User:", user_input, "| Intent:", intent)

    # store user message
    chat_memory[session_id].append(("User", user_input))

    # ---- Greeting ----
    if intent == "greeting":
        bot_response = "Hello 👋 I am your medical assistant. How can I help you today?"

    # ---- Identity ----
    elif intent == "identity":
        bot_response = "I am an AI medical assistant that answers health-related questions using medical knowledge documents."

    # ---- Small Talk ----
    elif intent == "chat":
        bot_response = "Please ask a medical or health-related question so I can assist you better."

    # ---- Symptom Follow-up ----
    elif intent == "symptom":
        bot_response = (
            "I understand you're experiencing symptoms.\n"
            "Since when has this started?\n"
            "Do you also have fever, vomiting, dizziness, or weakness?"
        )

    # ---- Medical RAG with Memory ----
    else:
        history_context = build_memory_context(session_id)

        enhanced_query = f"""
Conversation so far:
{history_context}

Current question:
{user_input}
"""

        response = rag_chain.invoke({"input": enhanced_query})
        bot_response = response["answer"]

        # limit long textbook outputs
        if len(bot_response.split()) > 120:
            bot_response = " ".join(bot_response.split()[:120]) + "..."

    # store bot reply
    chat_memory[session_id].append(("Assistant", bot_response))

    print("Bot:", bot_response)
    return bot_response


# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

