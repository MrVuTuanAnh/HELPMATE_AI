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
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import openai

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

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

client = PersistentClient(path='./chromadb/')
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Retrieve or create the main and cache collections
insurance_collection = client.get_or_create_collection(name='RAG_on_Insurance', embedding_function=embedding_function)
cache_collection = client.get_or_create_collection(name='Insurance_Cache', embedding_function=embedding_function)

# CrossEncoder initialization
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

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
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_no, tables, table_bboxes = f"Page {page.page_number}", page.find_tables(), [table.bbox for table in page.find_tables()]
            tables = [{'table': table.extract(), 'top': table.bbox[1]} for table in tables]
            non_table_words = [word for word in page.extract_words() if not any(check_bboxes(word, bbox) for bbox in table_bboxes)]
            lines = [' '.join(word['text'] for word in cluster) if 'text' in cluster[0] else json.dumps(cluster[0]['table'])
                     for cluster in pdfplumber.utils.cluster_objects(non_table_words + tables, itemgetter('top'), tolerance=5)]
            full_text.append([page_no, " ".join(lines)])
    return full_text

# PDF Processing and Data Preparation
url = "https://cdn.upgrad.com/uploads/production/585ca56a-6fe1-4b93-903c-1c1a1de74bf1/Principal-Sample-Life-Insurance-Policy.pdf"
save_path = './data/Principal-Sample-Life-Insurance-Policy.pdf'
download_pdf(url, save_path)

data = [pd.DataFrame(extract_text_from_pdf(save_path), columns=['Page No.', 'Page_Text']).assign(**{'Document Name': Path(save_path).name})]

# Data Manipulation and Analysis
if data:
    insurance_pdfs_data = pd.concat(data, ignore_index=True).loc[lambda df: df['Page_Text'].str.split().str.len() >= 10]
    insurance_pdfs_data['Metadata'] = insurance_pdfs_data.apply(lambda row: {'Policy_Name': row['Document Name'][:-4], 'Page_No.': row['Page No.']}, axis=1)
    print(insurance_pdfs_data.head())

# ChromaDB Integration for Embeddings
insurance_collection.add(documents=insurance_pdfs_data["Page_Text"].tolist(), ids=[str(i) for i in range(len(insurance_pdfs_data))],
                         metadatas=insurance_pdfs_data['Metadata'].tolist())

class UserQuery(BaseModel):
    query: str

@app.post("/query/")
async def query_endpoint(user_query: UserQuery):
    query = user_query.query
    # Similar query handling and response generation as previously described

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
