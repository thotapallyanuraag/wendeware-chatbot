import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Load OpenAI API key from secrets
api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key

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

# UI
st.title("🔌 Wendeware Support Chatbot")
st.write("Bitte stellen Sie Ihre Frage. Der Bot antwortet formal basierend auf offiziellen Wendeware-Inhalten.")
question = st.text_input("Ihre Frage:")

if question:
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    llm = ChatOpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=question)
    st.success(answer)
