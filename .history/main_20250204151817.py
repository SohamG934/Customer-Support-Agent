import os
import smtplib
from email.mime.text import MIMEText
import streamlit as st
import openai
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

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
def initialize_vectorstore(docs, vectorstore_path="faiss_index"):
    # For simplicity, we're skipping vectorstore initialization here, as we're no longer using it
    return docs

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
docs = initialize_vectorstore(docs)

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Type your question here:")

if user_input:
    greeting.empty()  # Remove greeting message
    
    # Limit chat history to last 6 conversations
    st.session_state.chat_history.append(f"User: {user_input}")
    if len(st.session_state.chat_history) > 12:  # 6 exchanges (user + agent)
        st.session_state.chat_history = st.session_state.chat_history[-12:]
    
    # Combine the last 6 conversations for context
    context = "\n".join(st.session_state.chat_history[-12:])

    # Create the prompt
    prompt = f"You are a helpful customer support assistant. Respond in {language}. Context: {context}. User's question: {user_input}"

    # Get response from OpenAI API
    response = openai.Completion.create(
        model="gpt-40-mini",  # or the model of your choice
        prompt=prompt,
        max_tokens=150,
        temperature=0.5
    ).choices[0].text.strip()
    
    # Append the agent's response to the conversation history
    st.session_state.chat_history.append(f"Agent: {response}")
    
    st.write("Agent:", response)
    
    # Detect dissatisfaction (simple heuristic check)
    if "I don't understand" in response or "not helpful" in response:
        if st.button("Submit Grievance"):
            user_email = st.text_input("Enter your email for grievance submission:")
            if user_email:
                send_grievance_email(user_email, f"User asked: {user_input}\nAgent response: {response}")
