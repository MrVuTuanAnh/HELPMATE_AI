from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import fitz  # PyMuPDF
import re
import pickle
import numpy as np
import faiss
from scipy.spatial.distance import cosine
import traceback
import os
from pathlib import Path
import requests
import pdfplumber
from operator import itemgetter
import json
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, SentenceTransformerEmbeddingFunction
from chromadb.utils import embedding_functions


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your environment.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def download_pdf(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Tệp đã được tải xuống và lưu tại: {save_path}")
    except requests.RequestException as e:
        print(f"Không thể tải xuống tệp: {e}")

url = "https://cdn.upgrad.com/uploads/production/585ca56a-6fe1-4b93-903c-1c1a1de74bf1/Principal-Sample-Life-Insurance-Policy.pdf"
save_path = './data/Principal-Sample-Life-Insurance-Policy.pdf'
download_pdf(url, save_path)

def check_bboxes(word, table_bbox):
    l = word['x0'], word['top'], word['x1'], word['bottom']
    r = table_bbox
    return l[0] > r[0] and l[1] > r[1] and l[2] < r[2] and l[3] < r[3]

def extract_text_from_pdf(pdf_path):
    p = 0
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_no = f"Page {p+1}"
            tables = page.find_tables()
            table_bboxes = [i.bbox for i in tables]
            tables = [{'table': i.extract(), 'top': i.bbox[1]} for i in tables]
            non_table_words = [word for word in page.extract_words() if not any(
                [check_bboxes(word, table_bbox) for table_bbox in table_bboxes])]
            lines = []
            for cluster in pdfplumber.utils.cluster_objects(non_table_words + tables, itemgetter('top'), tolerance=5):
                if 'text' in cluster[0]:
                    try:
                        lines.append(' '.join([i['text'] for i in cluster]))
                    except KeyError:
                        pass
                elif 'table' in cluster[0]:
                    lines.append(json.dumps(cluster[0]['table']))
            full_text.append([page_no, " ".join(lines)])
            p +=1
    return full_text

pdf_directory = Path("./data/")
data = []

for pdf_path in pdf_directory.glob("*.pdf"):
    print(f"...Processing {pdf_path.name}")
    extracted_text = extract_text_from_pdf(pdf_path)
    extracted_text_df = pd.DataFrame(extracted_text, columns=['Page No.', 'Page_Text'])
    extracted_text_df['Document Name'] = pdf_path.name
    data.append(extracted_text_df)
    print(f"Finished processing {pdf_path.name}")

print("All PDFs have been processed.")

if data:
    insurance_pdfs_data = pd.concat(data, ignore_index=True)
    insurance_pdfs_data['Text_Length'] = insurance_pdfs_data['Page_Text'].apply(lambda x: len(x.split(' ')))
    # Retain only the rows with a text length of at least 10
    insurance_pdfs_data = insurance_pdfs_data.loc[insurance_pdfs_data['Text_Length'] >= 10]
    
    # Store the metadata for each page in a separate column
    insurance_pdfs_data['Metadata'] = insurance_pdfs_data.apply(
        lambda x: {'Policy_Name': x['Document Name'][:-4], 'Page_No.': x['Page No.']},
        axis=1
    )
    
    if not insurance_pdfs_data.empty:
        print(f"Maximum text length: {max(insurance_pdfs_data['Text_Length'])}")
        print(insurance_pdfs_data.head())
    else:
        print("No data extracted from PDFs after filtering by text length.")
else:
    print("No PDFs were processed.")

# Import the OpenAI Embedding Function into chroma
# Define the path where chroma collections will be stored
# chroma_data_path = './chromadb/'
chroma_data_path = './chromadb/'
# Call PersistentClient()
client = chromadb.PersistentClient(path=chroma_data_path)
# Set up the embedding function using the OpenAI embedding model
model = "text-embedding-ada-002"
embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=model)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# Convert the page text and metadata from your dataframe to lists to be able to pass it to chroma
documents_list = insurance_pdfs_data["Page_Text"].tolist()
metadata_list = insurance_pdfs_data['Metadata'].tolist()
# insurance_collection = client.get_or_create_collection(name='RAG_on_Insurance', embedding_function=embedding_function)
insurance_collection = client.get_or_create_collection(name='RAG_on_Insurance', embedding_function=sentence_transformer_ef)
# Add the documents and metadata to the collection alongwith generic integer IDs. You can also feed the metadata information as IDs by combining the policy name and page no.
insurance_collection.add(
    documents= documents_list,
    ids = [str(i) for i in range(0, len(documents_list))],
    metadatas = metadata_list
)
# Let's take a look at the first few entries in the collection
insurance_collection.get(
    ids = ['0','1','2'],
    include = ['embeddings', 'documents', 'metadatas']
)