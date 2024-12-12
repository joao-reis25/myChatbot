from pdf_processor import process_pdfs, create_vector_store
import shutil
import os

def retrain():
    # Directory paths
    pdf_dir = '/Users/joaoreis/Documents/GitHub/myChatbot/myPDF/'
    chroma_dir = './chromadb'
    
    # Remove existing vector store
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
    
    # Process PDFs and create new vector store
    pdf_texts, pdf_filenames = process_pdfs(pdf_dir)
    vectorstore = create_vector_store(pdf_texts, pdf_filenames)
    
    print("Retraining complete!")

if __name__ == "__main__":
    retrain()
