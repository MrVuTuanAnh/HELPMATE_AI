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
from scipy.spatial.distance import cosine
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)
from dotenv import load_dotenv
import os
import traceback

# Correct placement for environment variable loading
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your environment.")

# Correctly set the OpenAI API key

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

# Định nghĩa lại tất cả các hàm từ phase2.py ở đây
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
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
#
#@app.post("/ask_openai/")
#async def ask_openai(question: Question):
#    try:
#        openai_answer = openai.Completion.create(
#            engine="text-davinci-003", # You may choose the engine you prefer
#            prompt=question.question,
#            max_tokens=150
#        )
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))
#
#    return {"question": question.question, "answer": openai_answer['choices'][0]['text']}
#
@app.post("/ask_with_openai/")
async def ask_with_openai(question: Question):
    try:
        # Use openai module directly
        response = client.completions.create(model="text-davinci-003",
        prompt=question.question,
        max_tokens=150)
        answer_text = response.choices[0].text
        return {"question": question.question, "answer": answer_text}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)