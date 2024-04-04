# Import necessary libraries
import fitz  # PyMuPDF for PDF extraction
import re  # For regular expressions during text cleaning
from sentence_transformers import SentenceTransformer  # For embeddings

# Function to extract text from a PDF document
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to clean and preprocess text
def clean_text(text):
    # Example: Remove simple headers/footers (customize as needed)
    cleaned_text = re.sub(r'Page \d+ of \d+', '', text)
    # Add more cleaning steps here if necessary
    return cleaned_text

# Function to chunk text by paragraph
def chunk_by_paragraph(text):
    chunks = text.split('\n\n')  # Assuming paragraphs are separated by two newlines
    return [chunk for chunk in chunks if chunk.strip() != '']

# Function to embed text chunks using SentenceTransformers
def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

# Main execution starts here
if __name__ == "__main__":
    pdf_path = './Principal-Sample-Life-Insurance-Policy.pdf'  # Update this path to your PDF document
    document_text = extract_text(pdf_path)
    cleaned_text = clean_text(document_text)
    chunks = chunk_by_paragraph(cleaned_text)
    embeddings = embed_chunks(chunks)

    # embeddings now contains the vector representations of your text chunks
