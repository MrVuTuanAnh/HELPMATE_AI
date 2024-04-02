from chromadb_client import ChromaDB

# Initialize ChromaDB client
chromadb = ChromaDB()

# Select a collection
collection = chromadb.collection("insurance_policy")

# Assume `documents` is a list of strings (your processed documents)
# and `metadata` is a list of dictionaries with metadata for each document
# Example: documents = ["document 1 text", "document 2 text"]
# Example: metadata = [{"id": "doc1", "topic": "science"}, {"id": "doc2", "topic": "math"}]

# Add documents and metadata to the collection
for doc, meta in zip(documents, metadata):
    collection.add(documents=[doc], metadatas=[meta])
