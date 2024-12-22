# import os
# import logging
# import pickle
# import torch
# from dotenv import load_dotenv
# from azure.search.documents import SearchClient
# from azure.core.credentials import AzureKeyCredential
# from sentence_transformers import SentenceTransformer
# import faiss

# # Load environment variables
# load_dotenv()

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Azure Search configuration
# AZURE_SEARCH_ENDPOINT = "https://gptkb-jcgzeo3krxxra.search.windows.net"
# AZURE_SEARCH_KEY = "2AlK6WpXT16VPryZYgEnTohsHsnmMAvAj7Wa6Do9dCAzSeCk6q9d"
# AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-tested-data-12')

# # Load re-ranker model
# RE_RANKER_MODEL_NAME = os.getenv('RE_RANKER_MODEL_NAME')
# model = SentenceTransformer(RE_RANKER_MODEL_NAME)

# # Load FAISS index and document mapping
# FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'faiss_index.index')
# DOCUMENT_MAPPING_PATH = os.getenv('DOCUMENT_MAPPING_PATH', 'document_mapping.pkl')

# def load_faiss_index():
#     try:
#         index = faiss.read_index(FAISS_INDEX_PATH)
#         with open(DOCUMENT_MAPPING_PATH, 'rb') as f:
#             document_mapping = pickle.load(f)
#         return index, document_mapping
#     except Exception as e:
#         logging.error(f"Error loading FAISS index or document mapping: {e}")
#         return None, None

# def query_azure_search(query):
#     search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
#     try:
#         response = search_client.search(search_text=query)
#         results = [{"id": result["id"], "title": result["title"], "content": result["content"]} for result in response]
#         return results
#     except Exception as e:
#         logging.error(f"Error querying Azure Search: {e}")
#         return []

# def query_faiss(query, model, document_mapping):
#     query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()
#     index, _ = load_faiss_index()
    
#     distances, indices = index.search(query_embedding.reshape(1, -1), k=10)  # Get top 10 results
#     results = [{"id": document_mapping[idx], "distance": dist} for dist, idx in zip(distances[0], indices[0])]
#     return results

# def re_rank_results(query, azure_results):
#     azure_titles = [result['title'] for result in azure_results]
#     azure_embeddings = model.encode(azure_titles, convert_to_tensor=True)
    
#     query_embedding = model.encode(query, convert_to_tensor=True)
#     cosine_scores = torch.nn.functional.cosine_similarity(query_embedding, azure_embeddings)
    
#     scored_results = [{'result': result, 'score': score.item()} for result, score in zip(azure_results, cosine_scores)]
#     sorted_results = sorted(scored_results, key=lambda x: x['score'], reverse=True)
    
#     return sorted_results

# def perform_search(query):
#     # Check the approach
#     approach = os.getenv('APPROACH', 'cloud')

#     if approach == 'cloud':
#         logging.info(f"Querying Azure Search for: {query}")
#         azure_results = query_azure_search(query)
        
#         if azure_results:
#             logging.info("Re-ranking Azure results...")
#             ranked_results = re_rank_results(query, azure_results)
#             for result in ranked_results:
#                 logging.info(f"Result: {result['result']} with score: {result['score']}")
#         else:
#             logging.warning("No results found in Azure Search.")
    
#     else:  # on-prem
#         logging.info(f"Querying FAISS for: {query}")
#         index, document_mapping = load_faiss_index()
#         if index and document_mapping:
#             faiss_results = query_faiss(query, model, document_mapping)
#             for result in faiss_results:
#                 logging.info(f"FAISS Result: {result['id']} with distance: {result['distance']}")
#         else:
#             logging.warning("No results found in FAISS.")

# if __name__ == "__main__":
#     query = "Fundamentals of Machine Learning."
#     perform_search(query)





import os
import logging
import pickle
import numpy as np
import torch
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Azure Search configuration
AZURE_SEARCH_ENDPOINT = "https://gptkb-jcgzeo3krxxra.search.windows.net"
AZURE_SEARCH_KEY = "2AlK6WpXT16VPryZYgEnTohsHsnmMAvAj7Wa6Do9dCAzSeCk6q9d"
AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-tested-data-12')

# Load re-ranker model
RE_RANKER_MODEL_NAME = os.getenv('RE_RANKER_MODEL_NAME', 'sentence-transformers/all-mpnet-base-v2')
model = SentenceTransformer(RE_RANKER_MODEL_NAME)

# Load FAISS index and document mapping
FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'faiss_index.index')
DOCUMENT_MAPPING_PATH = os.getenv('DOCUMENT_MAPPING_PATH', 'document_mapping.pkl')

def load_faiss_index():
    """Load FAISS index and document mapping."""
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOCUMENT_MAPPING_PATH, 'rb') as f:
            document_mapping = pickle.load(f)
        return index, document_mapping
    except Exception as e:
        logging.error(f"Error loading FAISS index or document mapping: {e}")
        return None, None

def create_and_save_document_mapping(document_paths):
    """Create and save document embeddings and their mapping."""
    document_mapping = []
    embeddings = []

    for doc_path in document_paths:
        content = ""

        # Check the file extension to determine how to read the file
        if doc_path.endswith('.pdf'):
            with open(doc_path, 'rb') as f:  # Open in binary mode for PDF
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    content += page.extract_text() or ""  # Extract text from each page
        elif doc_path.endswith('.txt'):
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()  # Read text files as normal
        elif doc_path.endswith('.docx'):
            from docx import Document
            doc = Document(doc_path)
            content = '\n'.join([para.text for para in doc.paragraphs])  # Read DOCX content
        else:
            logging.warning(f"Unsupported file format for {doc_path}")
            continue  # Skip unsupported formats

        # Log content length
        logging.info(f"Loaded content for {doc_path}: {len(content)} characters.")

        # Create the embedding for the content
        embedding = model.encode(content, convert_to_tensor=True).cpu().numpy()

        # Check if embedding is valid
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            logging.error(f"Invalid embedding generated for document: {doc_path}")
            continue
        
        # Append valid embeddings and document mappings
        embeddings.append(embedding)
        document_mapping.append(os.path.basename(doc_path))

    if embeddings:
        # Create FAISS index and save it
        embeddings = np.array(embeddings)  # Convert to numpy array
        faiss_index = faiss.IndexFlatL2(embeddings.shape[1])  # Initialize FAISS index
        faiss_index.add(embeddings)  # Add all embeddings to the index
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)  # Save the FAISS index

        # Save the document mapping
        with open(DOCUMENT_MAPPING_PATH, 'wb') as f:
            pickle.dump(document_mapping, f)

        logging.info("FAISS index and document mapping saved successfully.")
    else:
        logging.error("No valid embeddings were created. FAISS index not saved.")

def query_azure_search(query):
    """Query Azure Search with the provided query."""
    search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
    try:
        response = search_client.search(search_text=query)
        results = [{"id": result["id"], "title": result["title"], "content": result["content"]} for result in response]
        return results
    except Exception as e:
        logging.error(f"Error querying Azure Search: {e}")
        return []

def query_faiss(query, model, document_mapping):
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()
    logging.info(f"Query embedding shape: {query_embedding.shape}")
    logging.info(f"Query embedding values: {query_embedding}")

    index, _ = load_faiss_index()
    if index is not None:
        distances, indices = index.search(query_embedding.reshape(1, -1), k=10)  # Get top 10 results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist < 1e38:  # Only include valid distances
                results.append({"id": document_mapping[idx], "distance": dist})
                logging.info(f"Distance: {dist}, Document ID: {document_mapping[idx]}")
        return results
    return []



def re_rank_results(query, azure_results):
    """Re-rank Azure Search results based on cosine similarity with the query."""
    azure_titles = [result['title'] for result in azure_results]
    azure_embeddings = model.encode(azure_titles, convert_to_tensor=True)

    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = torch.nn.functional.cosine_similarity(query_embedding, azure_embeddings)

    scored_results = [{'result': result, 'score': score.item()} for result, score in zip(azure_results, cosine_scores)]
    sorted_results = sorted(scored_results, key=lambda x: x['score'], reverse=True)

    return sorted_results

def perform_search(query):
    """Perform the search based on the specified approach (cloud or on-prem)."""
    # Check the approach
    approach = os.getenv('APPROACH', 'cloud')

    if approach == 'cloud':
        logging.info(f"Querying Azure Search for: {query}")
        azure_results = query_azure_search(query)

        if azure_results:
            logging.info("Re-ranking Azure results...")
            ranked_results = re_rank_results(query, azure_results)
            for result in ranked_results:
                logging.info(f"Result: {result['result']} with score: {result['score']}")
        else:
            logging.warning("No results found in Azure Search.")

    else:  # on-prem
        logging.info(f"Querying FAISS for: {query}")
        index, document_mapping = load_faiss_index()
        if index and document_mapping:
            faiss_results = query_faiss(query, model, document_mapping)
            for result in faiss_results:
                logging.info(f"FAISS Result: {result['id']} with distance: {result['distance']}")
        else:
            logging.warning("No results found in FAISS.")

if __name__ == "__main__":
    # List your document paths here
    document_paths = [
        "D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/TDP Plan.pdf",
        # Add more document paths as needed
    ]

    # Uncomment the line below to create and save document mapping
    # create_and_save_document_mapping(document_paths)

    query = "Fundamentals of Machine Learning."
    perform_search(query)
