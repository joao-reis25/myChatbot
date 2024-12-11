import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from ollama import Client
import streamlit as st
import time
import sys
from typing import List, Dict
from langdetect import detect

# Streamlit app
# Configure page layout - add this at the very top
st.set_page_config(
    page_title="Your Chatbot Friend",
    page_icon="🤖",
    layout="wide",  # or "centered"
    initial_sidebar_state="expanded"  # or "collapsed"
)

st.title("Your Chatbot Friend")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'max_history' not in st.session_state:
    st.session_state.max_history = 20  # Limit chat history

# Add temperature control in sidebar
st.sidebar.title("Chat Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
st.sidebar.button("Clear Chat History", on_click=lambda: st.session_state.clear())

# Modify the language selection to include both input and output languages
st.sidebar.subheader("Language Settings")
input_language = st.sidebar.radio(
    "Your input language:",
    options=['Auto-detect', 'English', 'Español', 'Français', 'Deutsch', 'Italiano', 'Português'],
    horizontal=True
)

output_language = st.sidebar.radio(
    "Preferred answer language:",
    options=['English', 'Español', 'Français', 'Deutsch', 'Italiano', 'Português'],
    horizontal=True
)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Initialize ChromaDB client and collection using LangChain's Chroma
vectorstore = Chroma(
    embedding_function=embedding_model,
    collection_name="pdf_documents",
    persist_directory="./chromadb"
)

# Initialize Ollama client
ollama_client = Client()

# Move the chat display before the input
chat_container = st.container()
with chat_container:
    # Display messages in reverse order (newest first)
    messages = st.session_state.chat_history[-st.session_state.max_history:]
    for message in (messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Move the user input after the chat display
user_question = st.text_input("Enter your question:")

if user_question:
    try:
        # Add to history after displaying
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Detect input language if auto-detect is selected
        detected_language = detect(user_question) if input_language == 'Auto-detect' else input_language
        
        # Translate user question to English if it's not in English
        if detected_language != 'en' and detected_language != 'English':
            with st.spinner('Translating your question...'):
                translation_prompt = f"Translate the following text to English, maintaining the same meaning: {user_question}. Provide only the translation without any additional text."
                translated_question = ollama_client.generate(
                    model="llama3.1",
                    prompt=translation_prompt,
                    options={"temperature": 0.3}
                )
                english_question = translated_question['response'].strip()
        else:
            english_question = user_question

        with st.spinner('Processing your request...'):
            system_prompt = """You are a helpful AI assistant. Provide accurate, helpful, and concise answers.
            If you're unsure about something, admit it. Base your answers on the provided context."""
            
            results = vectorstore.similarity_search(query=english_question, k=3)

            if results:
                context = "\n".join([result.page_content for result in results])
                prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {english_question}"

                # Get response in English first
                response = ollama_client.generate(
                    model="llama3.1",
                    prompt=prompt,
                    options={"temperature": temperature}
                )
                english_response = response['response']

                # Translate to desired language if not English
                if output_language != 'English':
                    with st.spinner(f'Translating to {output_language}...'):
                        translation_prompt = f"Translate the following text to {output_language}, maintaining the same tone and meaning: {english_response}. Provide only the translation without any additional text."
                        translated_response = ollama_client.generate(
                            model="llama3.1",
                            prompt=translation_prompt,
                            options={"temperature": 0.3}
                        )
                        final_response = translated_response['response']
                else:
                    final_response = english_response

                # Display only the desired language answer
                with st.chat_message("assistant"):
                    st.write(final_response)

                # Add response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_response
                })

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}. Please try again in a moment."
        st.error(error_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
