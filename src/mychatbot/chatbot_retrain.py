import os
from datetime import datetime
from fpdf import FPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def create_and_store_new_pdf(question: str, answer: str) -> None:
    """
    Creates a new PDF with the Q&A and updates the vector store.
    
    Args:
        question (str): The user's question
        answer (str): The AI's answer
    """
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add content
    pdf.cell(200, 10, txt="Question & Answer Document", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"Question: {question}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Answer: {answer}")
    
    # Create pdfs directory if it doesn't exist
    os.makedirs("myPDF", exist_ok=True)
    
    # Save PDF with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"myPDF/qa_{timestamp}.pdf"
    pdf.output(pdf_path)
    
    # Load and process the new PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    
    # Update vector store
    vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_name="pdf_documents",
        persist_directory="./chromadb"
    )
    
    # Add new documents to existing collection
    vectorstore.add_documents(chunks)
    vectorstore.persist()

def get_answer_from_vectorstore(question: str) -> str:
    """
    Retrieves an answer from the vector store for a given question.
    
    Args:
        question (str): The user's question
    
    Returns:
        str: The retrieved answer
    """
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    
    # Load existing vector store
    vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_name="pdf_documents",
        persist_directory="./chromadb"
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Initialize LLM and QA chain
    llm = OpenAI()  # or your preferred LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    # Get answer
    return qa_chain.run(question)
