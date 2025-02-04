import os
import smtplib
from email.mime.text import MIMEText
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load and process company documentation
def load_documents(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())
        if file.endswith(".pdf"):
            print(f"Loading file: {file}")
            loader = PDFMinerLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())
    return documents

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
    sender_email = "sohamghadge0903@gmail.com"
    receiver_email = "ghadgesoham934@gmail.com"
    subject = "Customer Grievance Report"
    
    msg = MIMEText(f"User Email: {user_email}\n\nIssue Summary:\n{grievance_summary}")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, "ygfy brpt tzta pbip")  # Securely handle credentials
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
vectorstore = initialize_vectorstore(docs)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_variables=["question", "language"]  # ✅ Allow multiple inputs
)
llm = ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini")
prompt_template = PromptTemplate(
    input_variables=["question", "chat_history", "context", "language"],  # Add "context"
    template="You are a helpful customer support assistant. Respond in {language}. Context: {context}. Previous conversation: {chat_history}. User's question: {question}."
)
chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory, combine_docs_chain_kwargs={"prompt": prompt_template})

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Type your question here:")

if user_input:
    greeting.empty()  # Remove greeting message
    response = chain.run({"question": user_input, "chat_history": st.session_state.chat_history, "language": language})
    st.session_state.chat_history.append((user_input, response))
    st.write("Agent:", response)
    
    # Detect dissatisfaction (simple heuristic check)
    if "I don't understand" in response or "not helpful" in response:
        if st.button("Submit Grievance"):
            user_email = st.text_input("Enter your email for grievance submission:")
            if user_email:
                send_grievance_email(user_email, f"User asked: {user_input}\nAgent response: {response}")
