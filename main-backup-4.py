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

# Import third-party libraries for PDF processing and AI operations
import fitz  # PyMuPDF
import re
import pickle
import numpy as np
import faiss
from scipy.spatial.distance import cosine
import traceback

# Import libraries for embeddings and database operations
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, SentenceTransformerEmbeddingFunction

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
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

# Utility Functions for processing
def download_pdf(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded and saved at: {save_path}")
    except requests.RequestException as e:
        print(f"Failed to download file: {e}")

def check_bboxes(word, table_bbox):
    l = word['x0'], word['top'], word['x1'], word['bottom']
    r = table_bbox
    return l[0] > r[0] and l[1] > r[1] and l[2] < r[2] and l[3] < r[3]

def extract_text_from_pdf(pdf_path):
    p = 0
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_no, tables, table_bboxes = f"Page {p+1}", page.find_tables(), [table.bbox for table in page.find_tables()]
            tables = [{'table': table.extract(), 'top': table.bbox[1]} for table in tables]
            non_table_words = [word for word in page.extract_words() if not any(check_bboxes(word, bbox) for bbox in table_bboxes)]
            lines = [' '.join(word['text'] for word in cluster) if 'text' in cluster[0] else json.dumps(cluster[0]['table']) 
                     for cluster in pdfplumber.utils.cluster_objects(non_table_words + tables, itemgetter('top'), tolerance=5)]
            full_text.append([page_no, " ".join(lines)])
            p += 1
    return full_text

# PDF Processing and Data Preparation
url = "https://cdn.upgrad.com/uploads/production/585ca56a-6fe1-4b93-903c-1c1a1de74bf1/Principal-Sample-Life-Insurance-Policy.pdf"
download_pdf(url, save_path='./data/Principal-Sample-Life-Insurance-Policy.pdf')

data = [pd.DataFrame(extract_text_from_pdf(pdf_path), columns=['Page No.', 'Page_Text']).assign(**{'Document Name': pdf_path.name})
        for pdf_path in Path("./data/").glob("*.pdf")]

# Data Manipulation and Analysis
if data:
    insurance_pdfs_data = pd.concat(data, ignore_index=True).loc[lambda df: df['Page_Text'].str.split().str.len() >= 10]
    insurance_pdfs_data['Metadata'] = insurance_pdfs_data.apply(lambda row: {'Policy_Name': row['Document Name'][:-4], 'Page_No.': row['Page No.']}, axis=1)
    print(insurance_pdfs_data.head())

# ChromaDB Integration for Embeddings
client = PersistentClient(path='./chromadb/')
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
insurance_collection = client.get_or_create_collection(name='RAG_on_Insurance', embedding_function=embedding_function)
insurance_collection.add(documents=insurance_pdfs_data["Page_Text"].tolist(), ids=[str(i) for i in range(len(insurance_pdfs_data))], 
                         metadatas=insurance_pdfs_data['Metadata'].tolist())
print(insurance_collection.get(ids=['0', '1', '2'], include=['embeddings', 'documents', 'metadatas']))

# Initialise a collection in chroma and pass the embedding_function to it so that it used OpenAI embeddings to embed the documents
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
cache_collection = client.get_or_create_collection(name='Insurance_Cache', embedding_function=embedding_function)
cache_collection.peek()

# the design considerations for creating a cache layer in the semantic search system.
# Documnet query --
# - first search in cache
#  - if find return top k closeset / chunks of k documents the results
#  - if not find search in main vector db
#    - store the new query in cache
#      - if similar query occurs in future, it's easy to return answer
#    - search and index on the main vector db and return top k closeset / chunks of k documents the results
# Read the user query

query = input()

# Searh the Cache collection first
# Query the collection against the user query and return the top 20 results

cache_results = cache_collection.query(
    query_texts=query,
    n_results=1
)

cache_results

results = insurance_collection.query(
query_texts=query,
n_results=10
)
results.items()

# Implementing Cache in Semantic Search

# Set a threshold for cache searchA
threshold = 0.2

ids = []
documents = []
distances = []
metadatas = []
results_df = pd.DataFrame()


# If the distance is greater than the threshold, then return the results from the main collection.

if cache_results['distances'][0] == [] or cache_results['distances'][0][0] > threshold:
      # Query the collection against the user query and return the top 10 results
      results = insurance_collection.query(
      query_texts=query,
      n_results=10
      )

      # Store the query in cache_collection as document w.r.t to ChromaDB so that it can be embedded and searched against later
      # Store retrieved text, ids, distances and metadatas in cache_collection as metadatas, so that they can be fetched easily if a query indeed matches to a query in cache
      Keys = []
      Values = []

      for key, val in results.items():
        if val is None:
          continue
        if key != 'embeddings':
          for i in range(10): # Top 10 variable, we can also put as 25 for top_n
            Keys.append(str(key)+str(i))
            Values.append(str(val[0][i]))


      cache_collection.add(
          documents= [query],
          ids = [query],  # Or if you want to assign integers as IDs 0,1,2,.., then you can use "len(cache_results['documents'])" as will return the no. of queries currently in the cache and assign the next digit to the new query."
          metadatas = dict(zip(Keys, Values))
      )

      print("Not found in cache. Found in main collection.")

      result_dict = {'Metadatas': results['metadatas'][0], 'Documents': results['documents'][0], 'Distances': results['distances'][0], "IDs":results["ids"][0]}
      results_df = pd.DataFrame.from_dict(result_dict)
      results_df


# If the distance is, however, less than the threshold, you can return the results from cache

elif cache_results['distances'][0][0] <= threshold:
      cache_result_dict = cache_results['metadatas'][0][0]

      # Loop through each inner list and then through the dictionary
      for key, value in cache_result_dict.items():
          if 'ids' in key:
              ids.append(value)
          elif 'documents' in key:
              documents.append(value)
          elif 'distances' in key:
              distances.append(value)
          elif 'metadatas' in key:
              metadatas.append(value)

      print("Found in cache!")

      # Create a DataFrame
      results_df = pd.DataFrame({
        'IDs': ids,
        'Documents': documents,
        'Distances': distances,
        'Metadatas': metadatas
      })

      results_df

# Read the user query
query2 = input()
# Searh the Cache collection first
# Query the collection against the user query and return the top 20 results

cache_results2 = cache_collection.query(
    query_texts=query2,
    n_results=1
)
cache_results2

# Implementing Cache in Semantic Search

# Set a threshold for cache search
threshold = 0.2

ids2 = []
documents2 = []
distances2 = []
metadatas2 = []
results_df2 = pd.DataFrame()


# If the distance is greater than the threshold, then return the results from the main collection.

if cache_results2['distances'][0] == [] or cache_results2['distances'][0][0] > threshold:
      # Query the collection against the user query and return the top 10 results
      results = insurance_collection.query(
      query_texts=query2,
      n_results=10
      )

      # Store the query in cache_collection as document w.r.t to ChromaDB so that it can be embedded and searched against later
      # Store retrieved text, ids, distances and metadatas in cache_collection as metadatas, so that they can be fetched easily if a query indeed matches to a query in cache
      Keys2 = []
      Values2 = []

      for key, val in results.items():
        if val is None:
          continue
        if key != 'embeddings':
          for i in range(10): # Top 10 variable, we can also put as 25 for top_n
            Keys2.append(str(key)+str(i))
            Values2.append(str(val[0][i]))


      cache_collection.add(
          documents= [query2],
          ids = [query2],  # Or if you want to assign integers as IDs 0,1,2,.., then you can use "len(cache_results['documents'])" as will return the no. of queries currently in the cache and assign the next digit to the new query."
          metadatas = dict(zip(Keys2, Values2))
      )

      print("Not found in cache. Found in main collection.")

      result_dict2 = {'Metadatas': results['metadatas'][0], 'Documents': results['documents'][0], 'Distances': results['distances'][0], "IDs":results["ids"][0]}
      results_df2 = pd.DataFrame.from_dict(result_dict2)
      results_df2


# If the distance is, however, less than the threshold, you can return the results from cache

elif cache_results2['distances'][0][0] <= threshold:
      cache_result_dict2 = cache_results2['metadatas'][0][0]

      # Loop through each inner list and then through the dictionary
      for key, value in cache_result_dict2.items():
          if 'ids' in key:
              ids2.append(value)
          elif 'documents' in key:
              documents2.append(value)
          elif 'distances' in key:
              distances2.append(value)
          elif 'metadatas' in key:
              metadatas2.append(value)

      print("Found in cache!")

      # Create a DataFrame
      results_df2 = pd.DataFrame({
        'IDs': ids2,
        'Documents': documents2,
        'Distances': distances2,
        'Metadatas': metadatas2
      })

results_df2

# Read the user query

query3 = input()
# Searh the Cache collection first
# Query the collection against the user query and return the top 20 results

cache_results3 = cache_collection.query(
    query_texts=query3,
    n_results=1
)
cache_results3

# Implementing Cache in Semantic Search

# Set a threshold for cache search
threshold = 0.2

ids3 = []
documents3 = []
distances3 = []
metadatas3 = []
results_df3 = pd.DataFrame()


# If the distance is greater than the threshold, then return the results from the main collection.

if cache_results3['distances'][0] == [] or cache_results3['distances'][0][0] > threshold:
      # Query the collection against the user query and return the top 10 results
      results = insurance_collection.query(
      query_texts=query3,
      n_results=10
      )

      # Store the query in cache_collection as document w.r.t to ChromaDB so that it can be embedded and searched against later
      # Store retrieved text, ids, distances and metadatas in cache_collection as metadatas, so that they can be fetched easily if a query indeed matches to a query in cache
      Keys3 = []
      Values3 = []

      for key, val in results.items():
        if val is None:
          continue
        if key != 'embeddings':
          for i in range(10): # Top 10 variable, we can also put as 25 for top_n
            Keys3.append(str(key)+str(i))
            Values3.append(str(val[0][i]))


      cache_collection.add(
          documents= [query3],
          ids = [query3],  # Or if you want to assign integers as IDs 0,1,2,.., then you can use "len(cache_results['documents'])" as will return the no. of queries currently in the cache and assign the next digit to the new query."
          metadatas = dict(zip(Keys3, Values3))
      )

      print("Not found in cache. Found in main collection.")

      result_dict3 = {'Metadatas': results['metadatas'][0], 'Documents': results['documents'][0], 'Distances': results['distances'][0], "IDs":results["ids"][0]}
      results_df3 = pd.DataFrame.from_dict(result_dict3)
      results_df3


# If the distance is, however, less than the threshold, you can return the results from cache

elif cache_results3['distances'][0][0] <= threshold:
      cache_result_dict3 = cache_results3['metadatas'][0][0]

      # Loop through each inner list and then through the dictionary
      for key, value in cache_result_dict3.items():
          if 'ids' in key:
              ids3.append(value)
          elif 'documents' in key:
              documents3.append(value)
          elif 'distances' in key:
              distances3.append(value)
          elif 'metadatas' in key:
              metadatas3.append(value)

      print("Found in cache!")

      # Create a DataFrame
      results_df3 = pd.DataFrame({
        'IDs': ids3,
        'Documents': documents3,
        'Distances': distances3,
        'Metadatas': metadatas3
      })

 results_df3
     
