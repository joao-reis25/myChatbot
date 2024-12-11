import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from ollama import Client
import streamlit as st
import time

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

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.write("**You**:\n" + message["content"] + "\n")
    else:
        st.write("**Bot**:\n" + message["content"] + "\n")
        st.write("--------------------------------")

if user_question:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    
    # Show loading spinner
    with st.spinner('Thinking...'):
        try:
            # Query ChromaDB using the proper LangChain API
            results = vectorstore.similarity_search(query=user_question, k=3)

            if results:
                prompt = f"You are an assistant. Use the following documents to answer the question:\n"
                for result in results:
                    prompt += f"{result.page_content}\n"
                prompt += f"\nAnswer the following question: {user_question}"

                # Get the response from the Ollama model
                response = ollama_client.generate(model="llama3.1", prompt=prompt)
                
                # Check the language of the prompt first
                prompt_language = ollama_client.generate(model="llama3.1", prompt=f"Detect the language of the following text: {user_question}")
                if prompt_language != language:
                    translated_response = ollama_client.generate(model="llama3.1", 
                        prompt=f"Translate the following text to {language} and present only the answer: {response['response']}")
                    if 'response' in translated_response:
                        final_response = translated_response['response']
                    else:
                        final_response = "Translation failed. Please try again."
                else:
                    final_response = response['response']
                
                # Add bot response to history
                st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                
                # Display the latest response
                st.markdown(f"**You**:<br> {user_question}", unsafe_allow_html=True)
                st.markdown(f"**Bot**:<br> {final_response}", unsafe_allow_html=True)
                st.write("--------------------------------")
            else:
                error_msg = "I couldn't find any relevant information to answer your question."
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.error(error_msg)
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.error(error_msg)
