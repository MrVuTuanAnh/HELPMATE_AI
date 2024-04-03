# Import các thư viện cần thiết
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer

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

# Hàm giả định để lưu dữ liệu vào ChromaDB
def save_to_chromadb(ids, metadatas, documents, embeddings=None, uris=None, data=None):
    # Xây dựng cấu trúc dữ liệu để lưu trữ
    storage_structure = {
        'ids': [ids],
        'distances': [[]],  # Không được tính toán hoặc cập nhật trong ví dụ này
        'metadatas': [metadatas],
        'embeddings': embeddings,  # None trong trường hợp này
        'documents': [documents],
        'uris': uris,  # None trong trường hợp này
        'data': data  # None trong trường hợp này
    }
    # Thực hiện lưu dữ liệu vào ChromaDB
    # Đây chỉ là một hàm giả định, bạn cần thực hiện tương tác thực tế với ChromaDB ở đây
    pass

# Hàm chính để xử lý tài liệu
def process_document(pdf_path, chunk_size=500, model_name='all-MiniLM-L6-v2'):
    text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(text)
    chunks = chunk_text(cleaned_text, chunk_size)
    embeddings = embed_text(chunks, model_name)

    ids = list(range(len(chunks)))
    metadatas = [{"chunk_id": i} for i in ids]
    documents = chunks
    # Trong trường hợp này, embeddings được thiết lập là None khi lưu vào cấu trúc dữ liệu
    save_to_chromadb(ids, metadatas, documents)

    return chunks, embeddings

if __name__ == "__main__":
    pdf_path = "./Principal-Sample-Life-Insurance-Policy.pdf"
    chunks, embeddings = process_document(pdf_path)
    print(chunks[:2])
    print(embeddings[:2])
