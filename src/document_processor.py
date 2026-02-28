# Optimized version
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import os

def load_and_process_pdf(pdf_path: str) -> list[Document]:
    """
    Load a PDF using PyMuPDF, tag each page with [Page N] metadata,
    clean the text, and chunk it using LangChain.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
    documents = []
    
    # Open the PDF using PyMuPDF and extract text per page
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        # Clean text
        # Remove "Page X of Y" or just "Page X"
        text = re.sub(r'(?i)page\s+\d+(\s+of\s+\d+)?', '', text)
        # Remove standalone numbers that might be loose page numbers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove URLs
        text = re.sub(r'https?://[^\s]+', '', text)
        # Remove multiple whitespaces/newlines and replace with a single space or newline appropriately
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove non-ASCII characters
        text = text.encode('ascii', 'ignore').decode('utf-8')
        
        # It's good practice to strip leading/trailing whitespace
        text = text.strip()
        
        if text:  # Only add if there's actual text on the page
            # Tag with metadata as requested
            metadata = {
                "page": page_num + 1,  # 1-indexed for user readability
                "source": f"[Page {page_num + 1}]"
            }
            documents.append(Document(page_content=text, metadata=metadata))
            
    doc.close()

    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Deduplication logic
    unique_chunks = []
    seen_content = set()
    
    for chunk in chunks:
        content = chunk.page_content.strip()
        # Skip very short chunks
        if len(content) < 50:
            continue
            
        # Skip duplicates
        if content not in seen_content:
            seen_content.add(content)
            unique_chunks.append(chunk)
            
    return unique_chunks
