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

# Initialize FastAPI app and CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Question(BaseModel):
    question: str

# Utility functions
def check_bboxes(word, table_bbox):
    # Check whether word is inside a table bbox.
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
                    lines.append(' '.join([i['text'] for i in cluster]))
                elif 'table' in cluster[0]:
                    lines.append(json.dumps(cluster[0]['table']))
            full_text.append([page_no, " ".join(lines)])
            p += 1
    return full_text

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    text = re.sub(' +', ' ', text)
    return text

def chunk_text(text, chunk_size=500):
    words = text.split(' ')
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def embed_text(text_chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def search_faiss(index, query_embedding, k=1):
    D, I = index.search(np.array([query_embedding]).astype('float32').reshape(1, -1), k)
    return I[0]

def save_cache(data, file_name='cache.pkl'):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def load_cache(file_name='cache.pkl'):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# API endpoints Local
@app.post("/ask/")
async def ask(question: Question):
    cache = load_cache()
    if not cache:
        # Thực hiện xử lý tài liệu và lưu vào cache nếu cần
        pdf_path = "./Principal-Sample-Life-Insurance-Policy.pdf"  # Cập nhật đường dẫn file PDF
        text = extract_text_from_pdf(pdf_path)
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text)
        embeddings = np.array(embed_text(chunks))
        index = create_faiss_index(embeddings)
        save_cache({'chunks': chunks, 'embeddings': embeddings, 'index': index})
    else:
        chunks = cache['chunks']
        embeddings = cache['embeddings']
        index = create_faiss_index(np.array(embeddings))

    query_embedding = embed_text([question.question])
    closest_idx = search_faiss(index, query_embedding, 1)
    try:
        answer = chunks[closest_idx[0]]
    except IndexError:
        raise HTTPException(status_code=404, detail="Answer not found")
    return {"question": question.question, "answer": answer}

# API endpoints openai
@app.post("/ask_with_openai/")
async def ask_with_openai(question: Question):
    try:
        # Placeholder for OpenAI GPT-based answering
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.completions.create(model="gpt-3.5-turbo-instruct",
                                             prompt=question.question,
                                             max_tokens=150)
        answer_text = response.choices[0].text.strip()
        return {"question": question.question, "answer": answer_text}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Main function to run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
