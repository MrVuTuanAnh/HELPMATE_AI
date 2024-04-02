import fitz  # PyMuPDF

def clean_text(text):
    """
    Perform basic text cleaning. This function should be expanded based on the specific content of your PDF.
    For instance, you might want to remove specific lines that consistently appear as headers or footers.
    """
    # Split into lines and remove leading/trailing whitespace
    lines = text.strip().split('\n')
    # Remove empty lines
    lines = [line.strip() for line in lines if line.strip() != '']
    # Join the lines back together
    cleaned_text = '\n'.join(lines)
    return cleaned_text

def chunk_text(text, chunk_by='paragraph'):
    """
    Chunk text into sentences, paragraphs, or sections.
    This simple example focuses on paragraph chunking.
    """
    if chunk_by == 'paragraph':
        # Split the text into paragraphs (two or more newlines indicates a new paragraph)
        chunks = text.split('\n\n')
    else:
        # Additional chunking strategies should be implemented here
        chunks = [text]
    return chunks

def process_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    full_text = ""
    for page in doc:
        # Extract text from each page
        full_text += page.get_text()

    # Clean the text
    cleaned_text = clean_text(full_text)
    
    # Chunk the text into paragraphs (for this example)
    paragraphs = chunk_text(cleaned_text, chunk_by='paragraph')
    
    return paragraphs

# Path to your PDF file (corrected URL assignment)
pdf_path = './Principal-Sample-Life-Insurance-Policy.pdf'

# Process the PDF and get the paragraphs
paragraphs = process_pdf(pdf_path)

# Example: Print ALL paragraphs to check (uncomment the entire loop)
for i, paragraph in enumerate(paragraphs):
    print(f"Paragraph {i+1}:", paragraph, "\n")
