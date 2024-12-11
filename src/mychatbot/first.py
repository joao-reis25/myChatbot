# Import libraries
import os
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ollama import Client


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

# Store embeddings in ChromaDB using LangChain
vectorstore = Chroma(
    embedding_function=embedding_model,
    collection_name="pdf_documents",
    persist_directory="./chromadb"
)

# Add the documents to ChromaDB
for i, text in enumerate(pdf_texts):
    vectorstore.add_texts(texts=[text], ids=[pdf_filenames[i]])


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
user_question = "How is the thesis supervision done?"

# Query ChromaDB using the proper LangChain API
results = vectorstore.similarity_search(query=user_question, k=3)  # k=3 means top 3 results
# Query ChromaDB for relevant documents
# Example user question

# Process and generate a response using Ollama
if results:
    prompt = "You are an assistant. Use the following documents to answer the question:\n"
    for result in results:
        prompt += f"{result.page_content}\n"
    prompt += f"\nAnswer the following question: {user_question}"

    # Get the response from the Ollama model
    response = ollama_client.generate(model="llama3.1", prompt=prompt)
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
