import chromadb

# Hypothetical incorrect usage
settings = {"some_setting": "value"}
additional_params = {"settings": settings}
chroma_client = chromadb.Client("chromadb_index", settings=settings, **additional_params)
