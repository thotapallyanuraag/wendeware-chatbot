import streamlit as st
import random

# Define intents and responses
intents = {
    "greetings": {
        "patterns": ["hello", "hi", "hey"],
        "responses": ["Hello!", "Hi there!", "Hey!"]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you later"],
        "responses": ["See you later!", "Goodbye!", "Bye!"]
    }
}

# Function to match user input with intents
def match_intent(message):
    for intent, values in intents.items():
        for pattern in values["patterns"]:
            if pattern.lower() in message.lower():
                return intent
    return None

# Function to get response based on intent
def get_response(intent):
    if intent is not None:
        return random.choice(intents[intent]["responses"])
    else:
        return "I didn't understand that."

# Streamlit app
st.title("Chatbot")

user_input = st.text_input("You: ")

if user_input:
    intent = match_intent(user_input)
    response = get_response(intent)
    st.write("Bot: ", response)
