# myChatbot

A sophisticated chatbot system that leverages vector database technology to intelligently process and retrieve information from PDF documents.

## Features

- PDF Document Processing: Automatically extracts and processes text from PDF files
- Vector Database Integration: Utilizes vector embeddings for efficient information storage and retrieval
- Semantic Search: Performs context-aware searches to find relevant information
- Natural Language Processing: Understands and responds to user queries in a conversational manner

## How It Works

1. Document Ingestion: The system processes PDF documents and breaks them down into manageable chunks
2. Vector Embedding: Text chunks are converted into vector embeddings using advanced NLP models
3. Database Storage: Embeddings are stored in a vector database for efficient retrieval
4. Query Processing: User questions are processed and matched against stored information
5. Response Generation: Relevant information is retrieved and formatted into coherent responses

## Technical Stack

- Frontend Framework: Streamlit
- Vector Database: Chroma   
- Embedding Model: HuggingFaceEmbeddings
- PDF Processing: Langchain
- Backend Framework: Ollama

## Getting Started

1. Clone the repository
2. Install the dependencies:   ```bash
   pip install -r requirements.txt   ```
3. Make sure you have Ollama running locally

## Usage

1. Start the application:   ```bash
   streamlit run app.py   ```
2. Upload your PDF documents through the web interface
3. Ask questions about your documents in the chat interface
