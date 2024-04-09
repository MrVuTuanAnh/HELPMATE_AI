from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import fitz  # PyMuPDF
import re
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import concurrent.futures  # Để song song hóa việc trích xuất văn bản
from openai import OpenAI
import traceback

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your environment.")

# Setup OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

def load_cache():
    try:
        with open('cache.pkl', 'rb') as f:
            cache = pickle.load(f)
        return cache
    except FileNotFoundError:
        return None

def save_cache(cache):
    with open('cache.pkl', 'wb') as f:
        pickle.dump(cache, f)

def extract_text_from_pdf_concurrently(pdf_path):
    doc = fitz.open(pdf_path)
    text_parts = []

    def extract_text(page):
        return page.get_text()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        text_parts = list(executor.map(extract_text, [page for page in doc]))

    text = "".join(text_parts)
    return text

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
    """
    Search for the k-nearest neighbors in the FAISS index for a given query embedding.

    Parameters:
    - index: The FAISS index.
    - query_embedding: The embedding vector of the query.
    - k: The number of nearest neighbors to search for.

    Returns:
    - Indices of the k-nearest neighbors in the index.
    """
    D, I = index.search(np.array([query_embedding]), k)
    return I[0]

@app.post("/ask/")
async def ask(question: Question):
    cache = load_cache()
    if cache is None or 'index' not in cache:
        pdf_path = "./your_pdf_file.pdf"  # Update this path to your PDF file
        text = extract_text_from_pdf_concurrently(pdf_path)
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text)
        embeddings = embed_text(chunks)
        index = create_faiss_index(np.array(embeddings))
        cache = {'chunks': chunks, 'embeddings': embeddings, 'index': index}
        save_cache(cache)
    else:
        chunks = cache['chunks']
        embeddings = cache['embeddings']
        index = create_faiss_index(np.array(embeddings))

    query_embedding = embed_text([question.question])[0]
    closest_idx = search_faiss(index, query_embedding, 1)[0]
    try:
        answer = chunks[closest_idx]
    except IndexError:
        raise HTTPException(status_code=404, detail="Answer not found")
    return {"question": question.question, "answer": answer}

@app.post("/ask_with_openai/")
async def ask_with_openai(question: Question):
    try:
        response = client.completions.create(model="gpt-3.5-turbo-instruct",
                                             prompt=question.question,
                                             max_tokens=150)
        answer_text = response.choices[0].text.strip()
        return {"question": question.question, "answer": answer_text}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
