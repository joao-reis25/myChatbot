import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from ollama import Client
import streamlit as st
import time
import sys
from typing import List, Dict

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

# Example user question
user_question = st.text_input("Enter your question:")

# Select the language of the answer
language = st.radio(
    "Select the language for your answer:",
    options=['English', 'Español', 'Français', 'Deutsch', 'Italiano', 'Português'],
    horizontal=True
)

# Improve chat display with a container
# Move chat container outside of the user_question condition
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history[-st.session_state.max_history:]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if user_question:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_question)
    
    with st.spinner('Processing your request...'):
        try:
            # Improved prompt with system context
            system_prompt = """You are a helpful AI assistant. Provide accurate, helpful, and concise answers.
            If you're unsure about something, admit it. Base your answers on the provided context."""
            
            results = vectorstore.similarity_search(query=user_question, k=3)

            if results:
                context = "\n".join([result.page_content for result in results])
                prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {user_question}"

                # Add retry logic for API calls
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = ollama_client.generate(
                            model="llama3.1",
                            prompt=prompt,
                            options={
                                "temperature": temperature
                            }
                        )
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e
                        time.sleep(1)  # Wait before retry

                # Improved translation handling
                if language != 'English':
                    with st.spinner(f'Translating to {language}...'):
                        translation_prompt = f"Translate the following text to {language}, maintaining the same tone and meaning: {response['response']}. Do not include any other text or comments."
                        translated_response = ollama_client.generate(
                            model="llama3.1",
                            prompt=translation_prompt,
                            options={
                                "temperature": 0.3  # Lower temperature for translations
                            }
                        )
                        final_response = translated_response['response']
                else:
                    final_response = response['response']

                # Add response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_response
                })

                # Display latest response in chat container
                with st.chat_message("assistant"):
                    st.write(final_response)

            else:
                error_msg = "I couldn't find relevant information to answer your question. Could you please rephrase or ask something else?"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.error(error_msg)

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}. Please try again in a moment."
            st.error(error_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
