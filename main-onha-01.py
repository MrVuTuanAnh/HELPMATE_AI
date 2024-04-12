from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from pathlib import Path
import requests
import pdfplumber
from operator import itemgetter
import json
import pandas as pd
from dotenv import load_dotenv
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your environment.")

# Initialize FastAPI app with CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility Functions for processing
def download_pdf(url, save_path):
    """Downloads a PDF from a URL and saves it to the specified path."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded and saved at: {save_path}")
    except requests.RequestException as e:
        print(f"Failed to download file: {e}")

def check_bboxes(word, table_bbox):
    """Checks if a word's bounding box is within a table's bounding box."""
    l = word['x0'], word['top'], word['x1'], word['bottom']
    r = table_bbox
    return l[0] > r[0] and l[1] > r[1] and l[2] < r[2] and l[3] < r[3]

def extract_text_from_pdf(pdf_path):
    """Extracts and clusters text from a PDF, distinguishing between table and non-table content."""
    p = 0
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_no, tables, table_bboxes = f"Page {p+1}", page.find_tables(), [table.bbox for table in page.find_tables()]
            tables = [{'table': table.extract(), 'top': table.bbox[1]} for table in tables]
            non_table_words = [word for word in page.extract_words() if not any(check_bboxes(word, bbox) for bbox in table_bboxes)]
            lines = [' '.join(word['text'] for word in cluster) if 'text' in cluster[0] else json.dumps(cluster[0]['table']) 
                     for cluster in pdfplumber.utils.cluster_objects(non_table_words + tables, itemgetter('top'), tolerance=5)]
            full_text.append([page_no, " ".join(lines)])
            p += 1
    return full_text

# PDF Processing and Data Preparation
url = "https://cdn.upgrad.com/uploads/production/585ca56a-6fe1-4b93-903c-1c1a1de74bf1/Principal-Sample-Life-Insurance-Policy.pdf"
save_path = './data/Principal-Sample-Life-Insurance-Policy.pdf'
download_pdf(url, save_path)

data = [pd.DataFrame(extract_text_from_pdf(pdf_path), columns=['Page No.', 'Page_Text']).assign(**{'Document Name': pdf_path.name})
        for pdf_path in Path("./data/").glob("*.pdf")]

# Data Manipulation and Analysis
if data:
    insurance_pdfs_data = pd.concat(data, ignore_index=True).loc[lambda df: df['Page_Text'].str.split().str.len() >= 10]
    insurance_pdfs_data['Metadata'] = insurance_pdfs_data.apply(lambda row: {'Policy_Name': row['Document Name'][:-4], 'Page_No.': row['Page No.']}, axis=1)
    print(insurance_pdfs_data.head())

# ChromaDB Integration for Embeddings
client = PersistentClient(path='./chromadb/')
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
insurance_collection = client.get_or_create_collection(name='RAG_on_Insurance', embedding_function=embedding_function)
insurance_collection.add(documents=insurance_pdfs_data["Page_Text"].tolist(), ids=[str(i) for i in range(len(insurance_pdfs_data))], 
                         metadatas=insurance_pdfs_data['Metadata'].tolist())
print(insurance_collection.get(ids=['0', '1', '2'], include=['embeddings', 'documents', 'metadatas']))

# Retrieve or create the main collection where documents are stored
main_collection = client.get_or_create_collection(name='RAG_on_Insurance', embedding_function=embedding_function)

# Retrieve or create a cache collection to store recent queries and results
cache_collection = client.get_or_create_collection(name='Insurance_Cache', embedding_function=embedding_function)

# Function to handle the search process
def handle_search(query):
    # Step 1: Check if the query is in the cache
    cache_hit = cache_collection.query(query_texts=[query])  # Remove the 'top_k' parameter
    if cache_hit:
        # If cache_hit includes a 'results' key with a non-empty list, we have a hit
        if cache_hit.get('results', []):
            # Extract the results and manually handle top-k logic if necessary
            cached_results = cache_hit['results'][:1]  # Assuming results are sorted by relevance
            return cached_results

    # Step 2: If not in cache, search the main vector database
    search_results = main_collection.query(query_texts=[query])
    if search_results:
        # If search_results include a 'results' key with a non-empty list, we have search results
        if search_results.get('results', []):
            # Optionally, add the search query and results to the cache for future queries
            cache_collection.add_documents([query], search_results['results'][:1])  # Adjust as needed

            # Extract the results and manually handle top-k logic if necessary
            main_results = search_results['results'][:10]  # Adjust as needed
            return main_results

    # If we reach here, no results were found
    return None

# At the end of the script, you can add:
if __name__ == "__main__":
    # Allow the user to enter a search query
    user_query = input("Enter your search query: ")
    
    # Perform the search and handle results
    results = handle_search(user_query)
    
    # Check if any results were returned and print them
    if results:
        print("Search results found:")
        for result in results:
            print(result)
    else:
        print("No search results found.")