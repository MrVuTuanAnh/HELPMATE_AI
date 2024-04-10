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

# Load environment variables
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

# Data preparation
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

# Function to check whether a word is present in a table or not for segregation of regular text and tables
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
    # Check if the DataFrame is empty before sampling
    if not insurance_pdfs_data.empty:
        print(insurance_pdfs_data.sample(2))
    else:
        print("No data extracted from PDFs.")
else:
    print("No PDFs were processed.")

