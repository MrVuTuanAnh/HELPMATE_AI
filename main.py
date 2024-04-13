from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
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
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
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

##################
# GLOBAL VARIABLES

client = PersistentClient(path='./chromadb/')  # ChromaDB Integration for Embeddings
model = "text-embedding-ada-002"
embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=model)
sentence_transformer_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
cache_collection = client.get_or_create_collection(name='Insurance_Cache', embedding_function=embedding_function)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

#
#
#
# Utility Functions for processing
def download_pdf(url, save_path):
    print("=" * 20)
    print("==== check_bboxes ====")
    print("=" * 20)
    print()

    """Downloads a PDF from a URL and saves it to the specified path."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded and saved at: {save_path}")
    except requests.RequestException as e:
        print(f"Failed to download file: {e}")

def check_bboxes(word, table_bbox):
    print("=" * 20)
    print("==== check_bboxes ====")
    print("=" * 20)
    print()
        
    """Checks if a word's bounding box is within a table's bounding box."""
    l = word['x0'], word['top'], word['x1'], word['bottom']
    r = table_bbox
    return l[0] > r[0] and l[1] > r[1] and l[2] < r[2] and l[3] < r[3]

def extract_text_from_pdf(pdf_path):
    print("=" * 20)
    print("==== Extracting Text from PDF ====")
    print("=" * 20)
    print()

    """Extracts and clusters text from a PDF, distinguishing between table and non-table content."""
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



def main():
    # PDF Processing and Data Preparation
    url = "https://cdn.upgrad.com/uploads/production/585ca56a-6fe1-4b93-903c-1c1a1de74bf1/Principal-Sample-Life-Insurance-Policy.pdf"
    save_path = './data/Principal-Sample-Life-Insurance-Policy.pdf'
    download_pdf(url, save_path)

    data = [
        pd.DataFrame(extract_text_from_pdf(pdf_path), columns=['Page No.', 'Page_Text']).assign(**{'Document Name': pdf_path.name})
        for pdf_path in Path("./data/").glob("*.pdf")
    ]

    if not data:
        raise ValueError("No PDFs found in the data directory.")
    
    # Data Manipulation and Analysis
    insurance_pdfs_data = pd.concat(data, ignore_index=True).loc[lambda df: df['Page_Text'].str.split().str.len() >= 10]
    insurance_pdfs_data['Metadata'] = insurance_pdfs_data.apply(lambda row: {'Policy_Name': row['Document Name'][:-4], 'Page_No.': row['Page No.']}, axis=1)

    # ChromaDB Integration for Embeddings
    insurance_collection = client.get_or_create_collection(name='RAG_on_Insurance', embedding_function=sentence_transformer_ef)
    insurance_collection.add(
        documents=insurance_pdfs_data["Page_Text"].tolist(), 
        ids=[str(i) for i in range(len(insurance_pdfs_data))], 
        metadatas=insurance_pdfs_data['Metadata'].tolist()
    )

    def exec_query_search(query):
        if not query:
            raise ValueError("No query provided.")

        cache_results = cache_collection.query(query_texts=query, n_results=1)
        print("Raw cache results: ", cache_results)
        
        results = insurance_collection.query(query_texts=query, n_results=10)
        print("Raw results: ", results.items())
        
        threshold = 0.2
        
        ids = []
        documents = []
        distances = []
        metadatas = []
        results_df = pd.DataFrame()
        
        if cache_results['distances'][0] == [] or cache_results['distances'][0][0] > threshold:
            results = insurance_collection.query(query_texts=query, n_results=10)
        
            keys = []
            values = []
            
            for _key, _val in results.items():
                if _val is None:
                    continue
                
                if _key != 'embeddings':
                    for i in range(10):
                        keys.append(str(_key) + str(i))
                        values.append(str(_val[0][i]))
            
            cache_collection.add(
                documents=[query],
                ids=[query],
                metadatas=dict(zip(keys, values))
            )
        
            print("Not found in cache. Found in main collection.")
            result_dict = {'Metadatas': results['metadatas'][0], 'Documents': results['documents'][0], 'Distances': results['distances'][0], "IDs":results["ids"][0]}
            results_df = pd.DataFrame.from_dict(result_dict)
            print(f"Not found cache: \n {results_df}")
        elif cache_results['distances'][0][0] <= threshold:
            cache_result_dict = cache_results['metadatas'][0][0]
            
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
            results_df = pd.DataFrame({
                'IDs': ids,
                'Documents': documents,
                'Distances': distances,
                'Metadatas': metadatas
            })
            
            print(f"Found cache: \n {results_df}")
        return results_df
# Define the function to generate the response. Provide a comprehensive prompt that passes the user query and the top 3 results to the model

    def generate_response(query, top_3_RAG):
        """
        Generate a response using GPT-3.5's ChatCompletion based on the user query and retrieved information.
        """
        messages = [
                    {"role": "system", "content":  "You are a helpful assistant in the insurance domain who can effectively answer user queries about insurance policies and documents."},
                    {"role": "user", "content": f"""You are a helpful assistant in the insurance domain who can effectively answer user queries about insurance policies and documents.
                                                    You have a question asked by the user in '{query}' and you have some search results from a corpus of insurance documents in the dataframe '{top_3_RAG}'. These search results are essentially one page of an insurance document that may be relevant to the user query.

                                                    The column 'documents' inside this dataframe contains the actual text from the policy document and the column 'metadata' contains the policy name and source page. The text inside the document may also contain tables in the format of a list of lists where each of the nested lists indicates a row.

                                                    Use the documents in '{top_3_RAG}' to answer the query '{query}'. Frame an informative answer and also, use the dataframe to return the relevant policy names and page numbers as citations.

                                                    Follow the guidelines below when performing the task.
                                                    1. Try to provide relevant/accurate numbers if available.
                                                    2. You don’t have to necessarily use all the information in the dataframe. Only choose information that is relevant.
                                                    3. If the document text has tables with relevant information, please reformat the table and return the final information in a tabular in format.
                                                    3. Use the Metadatas columns in the dataframe to retrieve and cite the policy name(s) and page numbers(s) as citation.
                                                    4. If you can't provide the complete answer, please also provide any information that will help the user to search specific sections in the relevant cited documents.
                                                    5. You are a customer facing assistant, so do not provide any information on internal workings, just answer the query directly.

                                                    The generated response should answer the query directly addressing the user and avoiding additional information. If you think that the query is not relevant to the document, reply that the query is irrelevant. Provide the final response as a well-formatted and easily readable text along with the citation. Provide your complete response first with all information, and then provide the citations.
                                                    """},
                ]
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message.content.split('\n')
######
    query = input("Enter your query: (What are the default benefits and provisions of the Group Policy?)")
    results_df = exec_query_search(query)

    query2 = input("Enter your query: (What does it mean by 'the later of the Date of Issue'?)") 
    results_df2 = exec_query_search(query2)

    query3 = input("Enter your query: (What happens if a third-party service provider fails to provide the promised goods and services?)")
    results_df3 = exec_query_search(query3)

    # Re-Ranking with cross encoder
    # Test the cross encoder model
    scores = cross_encoder.predict(
        [
            ['Does the insurance cover diabetic patients?', 'The insurance policy covers some pre-existing conditions including diabetes, heart diseases, etc. The policy does not howev'],
            ['Does the insurance cover diabetic patients?', 'The premium rates for various age groups are given as follows. Age group (<18 years): Premium rate']
        ]
    )
    
    print(f'scores: {scores}')
<<<<<<< HEAD
    # Input (query, response) pairs for each of the top 20 responses received from the semantic search to the cross encoder
    # Generate the cross_encoder scores for these pairs

    cross_inputs = [[query, response] for response in results_df['Documents']]
    cross_rerank_scores = cross_encoder.predict(cross_inputs)

    # Store the rerank_scores in results_df
    results_df['Reranked_scores'] = cross_rerank_scores
=======
    
    # Input (query, response) pairs for each of the top 20 responses received from the semantic search to the cross encoder
    # Generate the cross_encoder scores for these pairs
    cross_inputs = [[query, response] for response in results_df['Documents']]
    cross_rerank_scores = cross_encoder.predict(cross_inputs)
    print(cross_rerank_scores)
    # Store the rerank_scores in results_df
    results_df['Reranked_scores'] = cross_rerank_scores
    print(results_df)
>>>>>>> 001ab7ee3aea4051f1d4178fc141bd3c04b5d92a
    # Return the top 3 results from semantic search
    top_3_semantic = results_df.sort_values(by='Distances')
    top_3_semantic[:3]
    # Return the top 3 results after reranking
    top_3_rerank = results_df.sort_values(by='Reranked_scores', ascending=False)
    top_3_rerank[:3]
<<<<<<< HEAD
    top_3_RAG_q1 = top_3_rerank[["Documents", "Metadatas"]][:3]
    print(top_3_RAG_q1)
    # .
=======
    print(top_3_rerank[:3])
    top_3_RAG_q1 = top_3_rerank[["Documents", "Metadatas"]][:3]
    print(top_3_RAG_q1)
    #### For 2nd Querry
    print(results_df2.head())
    query2 = input("Enter your query: (What does it mean by 'the later of the Date of Issue'?)") 
    # Input (query, response) pairs for each of the top 20 responses received from the semantic search to the cross encoder
    # Generate the cross_encoder scores for these pairs
    cross_inputs2 = [[query2, response] for response in results_df2['Documents']]
    cross_rerank_scores2 = cross_encoder.predict(cross_inputs2)
    print(cross_rerank_scores2)
    # Store the rerank_scores in results_df
    results_df2['Reranked_scores'] = cross_rerank_scores2
    print(results_df2)
    # Return the top 3 results from semantic search
    top_3_semantic2_q2 = results_df2.sort_values(by='Distances')
    top_3_semantic2_q2[:3]
    print(top_3_semantic2_q2[:3])
    # Return the top 3 results after reranking
    top_3_rerank_q2 = results_df2.sort_values(by='Reranked_scores', ascending=False)
    top_3_rerank_q2[:3]
    print(top_3_rerank_q2[:3])
    top_3_RAG_q2 = top_3_rerank_q2[["Documents", "Metadatas"]][:3]
    print(top_3_RAG_q2)
    ### For 3 query
    query3 = input("Enter your query: (What happens if a third-party service provider fails to provide the promised goods and services?)")
    results_df3.head()
    print(results_df3.head())
    # Input (query, response) pairs for each of the top 20 responses received from the semantic search to the cross encoder
    # Generate the cross_encoder scores for these pairs
    cross_inputs3 = [[query3, response] for response in results_df3['Documents']]
    cross_rerank_scores3 = cross_encoder.predict(cross_inputs3)
    cross_rerank_scores3
    print(cross_rerank_scores3)
    # Store the rerank_scores in results_df
    results_df3['Reranked_scores'] = cross_rerank_scores3
    results_df3
    print(results_df3)
    # Return the top 3 results from semantic search
    top_3_semantic_q3 = results_df3.sort_values(by='Distances')
    top_3_semantic_q3[:3]
    print(top_3_semantic_q3[:3])
    # Return the top 3 results after reranking
    top_3_rerank_q3 = results_df3.sort_values(by='Reranked_scores', ascending=False)
    top_3_rerank_q3[:3]
    print(top_3_rerank_q3[:3])
    top_3_RAG_q3 = top_3_rerank_q3[["Documents", "Metadatas"]][:3]
    top_3_RAG_q3
    print(top_3_RAG_q3)
    ##### retrieval Augmented Generation
    query = input("Enter your query: (What are the default benefits and provisions of the Group Policy?)")
    # Generate the response - For Query 1
    response = generate_response(query, top_3_RAG_q1)
    print("Query 1: ","\n",query,"\n_________________________________________________________________________________________________________________\n_________________________________________________________________________________________________________________\n")
    # Print the response
    print("\n".join(response))
    query2 = input("Enter your query: (What does it mean by 'the later of the Date of Issue'?)") 
    # Generate the response - For Query 2
    response2 = generate_response(query, top_3_RAG_q2)
    print("Query 2: ","\n",query2,"\n_________________________________________________________________________________________________________________\n_________________________________________________________________________________________________________________\n")
    # Print the response
    print("\n".join(response2))
    query3 = input("Enter your query: (What happens if a third-party service provider fails to provide the promised goods and services?)")
    # Generate the response - For Query 1
    response3 = generate_response(query, top_3_RAG_q3)
    print("Query 3: ","\n",query3,"\n_________________________________________________________________________________________________________________\n_________________________________________________________________________________________________________________\n")
    # Print the response
    print("\n".join(response3))
        # .
>>>>>>> 001ab7ee3aea4051f1d4178fc141bd3c04b5d92a
    # .
    # .
    # .

    

if __name__ == "__main__":
    main()
