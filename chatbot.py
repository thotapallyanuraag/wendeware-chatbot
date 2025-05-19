import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
import json

# Load API key from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Wendeware URLs
URLS = [
    "https://www.wendeware.com/",
    "https://www.wendeware.com/amperix-energiemanagementsystem",
    "https://www.wendeware.com/amperix-energiemanager",
    "https://www.wendeware.com/amperix-portal-mypowergrid",
    "https://www.wendeware.com/amperix-funktionen",
    "https://www.wendeware.com/rundsteuerempfaenger",
    "https://www.wendeware.com/dynamische-stromtarife",
    "https://www.wendeware.com/amperix-editionen",
    "https://www.wendeware.com/amperix-oem",
    "https://www.wendeware.com/ueber-uns",
    "https://www.wendeware.com/jobs",
    "https://www.wendeware.com/kontakt",
    "https://www.wendeware.com/produktsicherheit",
    "https://www.wendeware.com/service-und-support",
    "https://www.wendeware.com/kompatibilitaetsliste"
]

# Scrape website content
def fetch_website_text(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text.strip()
    except:
        return ""

@st.cache_data
def load_documents():
    docs = []
    for url in URLS:
        text = fetch_website_text(url)
        if text:
            docs.append(Document(page_content=text, metadata={"source": url}))
    return docs

# Session state to track conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("💬 Wendeware Chat Support")
st.markdown("Ask anything about the Amperix system. Answers are based on official Wendeware website content.")

user_input = st.chat_input("Type your message…")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    context = "\n\n".join([doc.page_content for doc in docs])[:3000]  # Limit context

    # Build message history for conversation
    messages = [{"role": "system", "content": "Answer the questions formally based on the provided context."}]
    for msg in st.session_state.chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages[-1]["content"] = f"Context: {context}\n\nQuestion: {user_input}"

    payload = {
        "model": "meta/llama3-8b-instruct",
        "messages": messages,
        "temperature": 0.5,
        "top_p": 1,
        "max_tokens": 1024,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            url="https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        result = response.json()
        answer = result['choices'][0]['message']['content']
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    except Exception as e:
        st.session_state.chat_history.append({"role": "assistant", "content": f"Request error: {e}"})

# Display conversation
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

