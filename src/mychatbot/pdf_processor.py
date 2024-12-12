import os
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def process_pdfs(pdf_directory):
    """Process all PDFs in the given directory and return texts and filenames."""
    pdf_texts = []
    pdf_filenames = []
    
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            pdf_text = extract_text_from_pdf(pdf_path)
            pdf_texts.append(pdf_text)
            pdf_filenames.append(pdf_file)
    
    return pdf_texts, pdf_filenames

def create_vector_store(pdf_texts, pdf_filenames, persist_directory="./chromadb"):
    """Create and persist the vector store from PDF texts."""
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    
    # Split texts into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Process chunks with metadata
    all_chunks = []
    for i, text in enumerate(pdf_texts):
        chunks = text_splitter.split_text(text)
        all_chunks.extend([{
            "content": chunk, 
            "source": pdf_filenames[i]
        } for chunk in chunks])
    
    # Create and persist the vector store
    texts = [chunk["content"] for chunk in all_chunks]
    metadatas = [{"source": chunk["source"]} for chunk in all_chunks]
    
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=persist_directory,
        collection_name="pdf_documents"
    )
    
    return vectorstore

def main():
    # Directory where PDF files are located
    pdf_dir = '/Users/joaoreis/Documents/GitHub/myChatbot/myPDF/'
    
    # Process PDFs
    pdf_texts, pdf_filenames = process_pdfs(pdf_dir)
    print(f"Processed {len(pdf_filenames)} documents:", pdf_filenames)
    
    # Create vector store
    vectorstore = create_vector_store(pdf_texts, pdf_filenames)
    print("Vector store created successfully!")

if __name__ == "__main__":
    main()
