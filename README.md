# HELPMATE_AI


git pull;git fetch;git add -A ;git commit -a -m "Update all files"; git push --all;


python -m venv helpmate_env

source helpmate_env/bin/activate

source helpmate_env/bin/deactived

pip install -r requirements.txt


Step 1: Install PyMuPDF
Make sure PyMuPDF is installed in your environment. If not, you can install it using pip:

sh
Copy code
pip install PyMuPDF


Step 2: Python Code for PDF Parsing and Chunking
python
Copy code
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

# Path to your PDF file
pdf_path = 'path/to/your/pdf/document.pdf'
paragraphs = process_pdf(pdf_path)

# Example: Print the first 5 paragraphs to check
for i, paragraph in enumerate(paragraphs[:5]):
    print(f"Paragraph {i+1}:", paragraph, "\n")






ERROR: Ignored the following versions that require a different python version: 1.21.2 Requires-Python >=3.7,<3.11; 1.21.3 Requires-Python >=3.7,<3.11; 1.21.4 Requires-Python >=3.7,<3.11; 1.21.5 Requires-Python >=3.7,<3.11; 1.21.6 Requires-Python >=3.7,<3.11
ERROR: Could not find a version that satisfies the requirement torch==1.11.0 (from versions: 1.13.0, 1.13.1, 2.0.0, 2.0.1, 2.1.0, 2.1.1, 2.1.2, 2.2.0, 2.2.1, 2.2.2)
ERROR: No matching distribution found for torch==1.11.0


