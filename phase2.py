import fitz  # PyMuPDF
import re
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from scipy.spatial.distance import cosine

# Hàm trích xuất văn bản từ PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Hàm làm sạch và tiền xử lý văn bản
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    text = re.sub(' +', ' ', text)
    return text

# Hàm chia nhỏ văn bản
def chunk_text(text, chunk_size=500):
    words = text.split(' ')
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Hàm nhúng văn bản
def embed_text(text_chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    return embeddings

# Hàm nhúng câu hỏi
def embed_query(query, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])
    return query_embedding[0]

# Tạo faiss index cho embeddings
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

# Tìm kiếm với faiss
def search_faiss(index, query_embedding, k=1):
    D, I = index.search(np.array([query_embedding]).astype('float32').reshape(1, -1), k)
    return I[0]

# Thêm cơ chế lưu trữ bộ nhớ đệm sử dụng pickle
def save_cache(data, file_name='cache.pkl'):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def load_cache(file_name='cache.pkl'):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def main(pdf_path, queries):
    # Kiểm tra cache trước
    cache = load_cache()
    if cache:
        chunks, embeddings = cache['chunks'], cache['embeddings']
    else:
        text = extract_text_from_pdf(pdf_path)
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text)
        embeddings = np.array(embed_text(chunks))
        # Lưu vào cache
        save_cache({'chunks': chunks, 'embeddings': embeddings})

    # Tạo index faiss từ embeddings
    index = create_faiss_index(embeddings)

    # Xử lý mỗi câu hỏi
    for query in queries:
        query_embedding = embed_query(query)
        closest_idx = search_faiss(index, query_embedding, k=1)
        print(f"Query: {query}\nClosest chunk: {chunks[closest_idx[0]]}\n")

if __name__ == "__main__":
    pdf_path = "./Principal-Sample-Life-Insurance-Policy.pdf"
    queries = [
        "What are the eligibility criteria for the insurance policy?",
        "How can one file a claim?",
        "What are the benefits included in the insurance plan?"
    ]
    main(pdf_path, queries)
