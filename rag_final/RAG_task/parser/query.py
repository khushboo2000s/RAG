import os
import logging
import pickle
import torch
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Azure Search configuration
AZURE_SEARCH_ENDPOINT = os.getenv('AZURE_SEARCH_ENDPOINT')
AZURE_SEARCH_KEY = os.getenv('AZURE_SEARCH_KEY')
AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-tested-data-seven')

# Load re-ranker model
RE_RANKER_MODEL_NAME = os.getenv('RE_RANKER_MODEL_NAME')
model = SentenceTransformer(RE_RANKER_MODEL_NAME)

# Load FAISS index and document mapping
FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'faiss_index.index')
DOCUMENT_MAPPING_PATH = os.getenv('DOCUMENT_MAPPING_PATH', 'document_mapping.pkl')

def load_faiss_index():
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOCUMENT_MAPPING_PATH, 'rb') as f:
            document_mapping = pickle.load(f)
        return index, document_mapping
    except Exception as e:
        logging.error(f"Error loading FAISS index or document mapping: {e}")
        return None, None

def query_azure_search(query):
    search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
    try:
        response = search_client.search(search_text=query)
        results = [{"id": result["id"], "title": result["title"], "content": result["content"]} for result in response]
        return results
    except Exception as e:
        logging.error(f"Error querying Azure Search: {e}")
        return []

def re_rank_results(query, azure_results):
    # Get embeddings for Azure results
    azure_titles = [result['title'] for result in azure_results]
    azure_embeddings = model.encode(azure_titles, convert_to_tensor=True)
    
    # Get embedding for the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Calculate similarity scores
    cosine_scores = torch.nn.functional.cosine_similarity(query_embedding, azure_embeddings)
    
    # Combine results and scores
    scored_results = [{'result': result, 'score': score.item()} for result, score in zip(azure_results, cosine_scores)]
    
    # Sort results by score
    sorted_results = sorted(scored_results, key=lambda x: x['score'], reverse=True)
    
    return sorted_results

def perform_search(query):
    logging.info(f"Querying Azure Search for: {query}")
    azure_results = query_azure_search(query)
    
    if azure_results:
        logging.info("Re-ranking results...")
        ranked_results = re_rank_results(query, azure_results)
        for result in ranked_results:
            logging.info(f"Result: {result['result']} with score: {result['score']}")
    else:
        logging.warning("No results found.")

if __name__ == "__main__":
    query = "Data Visualization Principles"
    perform_search(query)
