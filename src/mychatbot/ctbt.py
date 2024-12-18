import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from ollama import Client
import streamlit as st
import time
import sys
from typing import List, Dict
from langdetect import detect
import subprocess

# Streamlit app
# Configure page layout - add this at the very top
st.set_page_config(
    page_title="Your Law Chatbot",
    page_icon="🤖",
    layout="wide",  # or "centered"
    initial_sidebar_state="expanded"  # or "collapsed"
)

st.title("Your Law Chatbot")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'max_history' not in st.session_state:
    st.session_state.max_history = 20  # Limit chat history

# Add temperature control in sidebar
st.sidebar.title("Chat Settings")
#temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
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

# Create a main container with two rows
main_container = st.container()

# Handle any existing question before creating new widgets
if "user_input" in st.session_state and st.session_state["user_input"]:
    try:
        current_question = st.session_state["user_input"]
        st.session_state["user_input"] = ""  # Clear input
        
        # Move all processing inside a with st.empty(): block to prevent display
        with st.empty():
            detected_language = detect(current_question) if input_language == 'Auto-detect' else input_language
            
            st.session_state.chat_history.append({"role": "user", "content": current_question})
            
            # Translate user question to English if it's not in English
            if detected_language != 'en' and detected_language != 'English':
                with st.spinner('Translating your question...'):
                    translation_prompt = f"Translate the following text to English, maintaining the same meaning: {current_question}. Provide only the translation without any additional text."
                    translated_question = ollama_client.generate(
                        model="llama3.1",
                        prompt=translation_prompt,
                        options={"temperature": 0.0}
                    )
                    english_question = translated_question['response'].strip()
            else:
                english_question = current_question

            with st.spinner('Processing your request...'):
                system_prompt = """You are a helpful AI assistant specialized in law. Your primary task is to answer questions based on the provided PDF documents.
                If the question cannot be answered using the provided context, clearly state that the information is not found in the documents.
                Always reference the specific parts of the documents that support your answer. Include file transcript in your answer when possible"""
                
                results = vectorstore.similarity_search(query=english_question, k=3)

                if results:
                    # Format context with clear separation between different document sections
                    context_parts = []
                    for i, result in enumerate(results, 1):
                        context_parts.append(f"Document section {i}:\n{result.page_content}")
                    context = "\n\n".join(context_parts)
                    
                    prompt = f"""{system_prompt}

Context from PDF documents:
{context}

Question: {english_question}

Please answer based on the above context from the PDF documents. If the information isn't found in the context, please say so clearly."""

                    # Get response in English first
                    response = ollama_client.generate(
                        model="llama3.1",
                        prompt=prompt,
                        options={"temperature": 0.1}  # Lower temperature for more focused answers
                    )
                    english_response = response['response']

                    # Check if the response indicates no information
                    if "information is not available" in english_response.lower() or "don't have" in english_response.lower():
                        # Run the retraining script
                        try:
                            subprocess.run(["python", "chatbot_retrain.py"], check=True)
                            st.warning("Retraining model with updated information. Please ask your question again.")
                            time.sleep(2)  # Give some time for the warning to be visible
                            st.rerun()
                        except subprocess.CalledProcessError as e:
                            st.error(f"Error during retraining: {str(e)}")
                            
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

                    # After getting the final_response, store it
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": final_response
                    })
        
        st.rerun()

    except Exception as e:
        with st.empty():
            error_msg = f"An error occurred: {str(e)}. Please try again in a moment."
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            st.rerun()

# Now create the UI elements
with main_container:
    # Create a two-part layout
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Chat history area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    messages = st.session_state.chat_history[-st.session_state.max_history:]
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    st.markdown('<div class="input-area" style="height: 1pt; background-color: white;"></div>', unsafe_allow_html=True)
    user_question = st.text_input("Enter your question:", key="user_input")
    #st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
