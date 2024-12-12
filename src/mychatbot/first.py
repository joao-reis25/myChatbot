# Import libraries
import os
from PyPDF2 import PdfReader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ollama import Client
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Use PyPDF2 to extract text from the PDFs
# Function to extract text from a PDF file
# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text
    
# Directory where PDF files are located
pdf_dir = '/Users/joaoreis/Documents/GitHub/myChatbot/myPDF/'
pdf_texts = []
pdf_filenames = []

# Extract text from each PDF and store the filenames
for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        pdf_text = extract_text_from_pdf(pdf_path)
        pdf_texts.append(pdf_text)
        pdf_filenames.append(pdf_file)

print("Text extraction complete. Found documents:", pdf_filenames)

# Initialize the SentenceTransformer model
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Add after PDF text extraction
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

all_chunks = []
for i, text in enumerate(pdf_texts):
    chunks = text_splitter.split_text(text)
    # Add metadata to keep track of source document
    all_chunks.extend([{"content": chunk, "source": pdf_filenames[i]} for chunk in chunks])

# Modify the vectorstore creation
texts = [chunk["content"] for chunk in all_chunks]
metadatas = [{"source": chunk["source"]} for chunk in all_chunks]
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    metadatas=metadatas,
    persist_directory="./chromadb"
)

# Use LangChain’s RetrievalQA with Ollama
# We will use LangChain’s RetrievalQA chain and Ollama’s Client for querying the LLaMA3.1 model.
# Initialize the Ollama client

# Initialize Ollama client
ollama_client = Client()

# Function to query LLaMA3 via Ollama's Client
#def get_llama3_response(prompt):
#    response = ollama_client.generate(model="llama3.1", prompt=prompt)
#    return response['text']

# Example user question
user_question = ""

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Modify your query function to include conversation history
def get_response(question, memory):
    results = vectorstore.similarity_search(query=question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Include chat history in prompt
    chat_history = memory.chat_memory.messages
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
    
    prompt = f"""Previous conversation:
{history_str}

Context from documents:
{context}

Current question: {question}

Please provide a detailed answer based on the context provided."""

    response = ollama_client.generate(
        model="llama3.1",
        prompt=prompt,
        options={
            "temperature": 0.3,
            #"max_tokens": 500
        }
    )
    
    # Update memory
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(response['response'])
    
    return response['response']

# Query ChromaDB using the proper LangChain API
results = vectorstore.similarity_search(query=user_question, k=3)  # k=3 means top 3 results
# Query ChromaDB for relevant documents
# Example user question

def create_prompt(question, context):
    return f"""Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {question}

Answer:"""

# Process and generate a response using Ollama
if results:
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = create_prompt(user_question, context)
    
    response = ollama_client.generate(
        model="llama3.1",
        prompt=prompt,
        options={
            "temperature": 0.7,
            #"num_predict": 500  # replaces max_tokens for Ollama
        }
    )
    if 'response' in response:
        print(f"Answer: {response['response']}")
    else:
        print("No 'response' key found in the response.")
else:
    print("No relevant information found.")

# Check if documents are retrieved
if results:
    print("\nRetrieved documents:")
    for i, result in enumerate(results, 1):
        print(f"Document {i} content: {result.page_content[:200]}...")  # Print first 200 characters
else:
    print("No relevant documents found.")

print("\nPDF texts content:")
for i, text in enumerate(pdf_texts, 1):
    print(f"PDF {i} content: {text[:200]}...")  # Print first 200 characters


def get_relevant_documents(question, threshold=0.5):
    results = vectorstore.similarity_search_with_score(query=question, k=5)
    
    # Filter results based on similarity score
    relevant_docs = [
        doc for doc, score in results 
        if score >= threshold
    ]
    
    return relevant_docs
