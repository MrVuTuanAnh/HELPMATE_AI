import fitz  # PyMuPDF for PDF extraction
import re  # Regular expressions for text cleaning
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss  # For efficient similarity search

# Extract text from PDF
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Clean and preprocess text
def clean_text(text):
    cleaned_text = re.sub(r'Page \d+ of \d+', '', text)
    return cleaned_text

# Chunk text by paragraph
def chunk_by_paragraph(text):
    chunks = text.split('\n\n')  
    return [chunk for chunk in chunks if chunk.strip() != '']

# Embed text chunks
def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

# Initialize and train FAISS index
def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# Cache for search results
cache = {}

# Perform search with caching
def search_with_cache(index, query_embedding, k=5):
    key = tuple(query_embedding.flatten())
    if key in cache:
        print("Retrieving cached results")
        return cache[key]
    else:
        distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
        cache[key] = indices
        return indices

# Main execution
if __name__ == "__main__":
    # Example use
    pdf_path = './Principal-Sample-Life-Insurance-Policy.pdf'  # Update this path
    document_text = extract_text(pdf_path)
    cleaned_text = clean_text(document_text)
    chunks = chunk_by_paragraph(cleaned_text)
    chunk_embeddings = np.array(embed_chunks(chunks)).astype('float32')
    
    # Prepare FAISS index
    index = create_faiss_index(chunk_embeddings)
    
    # Test queries
    queries = ["What is the coverage period?", "How are claims processed?", "What exclusions apply?"]
    query_embeddings = np.array(embed_chunks(queries)).astype('float32')
    
    # Perform searches
    for query_embedding in query_embeddings:
        indices = search_with_cache(index, query_embedding)
        print("Top chunks for query:", indices)
