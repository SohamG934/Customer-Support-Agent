import os
import smtplib
from email.mime.text import MIMEText
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}



# Load and process company documentation
def load_documents(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r") as f:
                documents.append(f.read())
        if file.endswith(".pdf"):
            # Add your PDF loading logic here (use PyPDF2 or similar)
            pass
    return documents

# Initialize vector database only if not already present
# Initialize vector database only if not already present
def initialize_vectorstore(docs, vectorstore_path="faiss_index"):
    if os.path.exists(vectorstore_path):
        return FAISS.load_local(vectorstore_path, SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(vectorstore_path)
    return vectorstore

# Function to send grievance email
def send_grievance_email(user_email, grievance_summary):
    sender_email = user_email
    receiver_email = "ghadgesoham934@gmail.com"
    subject = "Customer Grievance Report"
    
    msg = MIMEText(f"User Email: {user_email}\n\nIssue Summary:\n{grievance_summary}")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.ehlo()
        server.login('sohamghadge0903@gmail.com', "ygfy brpt tzta pbip")  # Securely handle credentials
        print("logged into smtp server")
        server.sendmail(sender_email, receiver_email, msg.as_string())
    
    st.success("Your grievance has been submitted.")

# Set up Streamlit interface
st.set_page_config(page_title="Customer Support Agent", layout="wide")
st.title("Customer Support Chatbot")

# Greeting message
greeting = st.empty()
greeting.text("Hello! How may I assist you with your Samsung Device?")

# Select language
language = st.selectbox("Select your language:", ["English", "Japanese", "Spanish", "French", "German", "Chinese"])

# Load documents and initialize vectorstore
folder_path = "docs"
docs = load_documents(folder_path)
db = initialize_vectorstore(docs)

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Type your question here:")
docs = db.similarity_search(user_input, k=3)
context = docs[0].page_content

if user_input:
    greeting.empty()  # Remove greeting message
    
    # Limit chat history to last 6 conversations
    st.session_state.chat_history.append(f"User: {user_input}")
    if len(st.session_state.chat_history) > 12:  # 6 exchanges (user + agent)
        st.session_state.chat_history = st.session_state.chat_history[-12:]
    
    # Combine the last 6 conversations for context
    chat_history = "\n".join(st.session_state.chat_history[-12:])

    # Create the prompt
    prompt = f"You are a helpful customer support assistant. Respond in formal {language}. Please adhere to provided context, user's question and conversation history. Context: {context}. history: {chat_history}."

    model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction= prompt
            )
    # chat_session = model.start_chat(history=chat_history)

    # Get response from OpenAI API
    response = model.generate_content(f"Please follow the system instructions for the following query. User's question: {user_input}")
    
    # Append the agent's response to the conversation history
    st.session_state.chat_history.append(f"Agent: {response.text}")
    
    st.write("Agent:", response.text)
    
    # Detect dissatisfaction (simple heuristic check)
if "I don't understand" in user_input or "not helpful" in user_input:
        if st.button("Submit Grievance"):
            user_email = st.text_input("Enter your email for grievance submission:")
            send_grievance_email(user_email, f"User asked: {user_input}\nAgent response: {response}")
            st.write("Grievance notified to our support team. They will be in touch shortly. Happy to help in other queries, if any!")
