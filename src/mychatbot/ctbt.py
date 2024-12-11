import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from ollama import Client
import streamlit as st

# Streamlit app
# Configure page layout - add this at the very top
st.set_page_config(
    page_title="Your Chatbot Friend",
    page_icon="🤖",
    layout="wide",  # or "centered"
    initial_sidebar_state="expanded"  # or "collapsed"
)

st.title("Your Chatbot Friend")

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


if user_question:
    # Query ChromaDB using the proper LangChain API
    results = vectorstore.similarity_search(query=user_question, k=3)  # k=3 means top 3 results

    # Process and generate a response using Ollama
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
            translated_response = ollama_client.generate(model="llama3.1", prompt=f"Translate the following text to {language} and present only the answer: {response['response']}. Do not mention the translation in your response.")
            if 'response' in translated_response:
                st.write(f"{translated_response['response']}")
            else:
                st.write("Translation failed. No 'response' key found in the translation response.")
        else:
            st.write(f"**Answer:** {response['response']}")
        # if 'response' in response:
        #     st.write(f"**Answer:** {response['response']}")
        # else:
        #     st.write("No 'response' key found in the response.")
    else:
        st.write("No relevant information found.")
