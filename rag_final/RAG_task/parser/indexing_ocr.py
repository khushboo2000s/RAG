import os
import logging
import numpy as np
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchableField,
    SearchField
)
from config import Config
from dotenv import load_dotenv
import faiss
import pickle
import re
from search_utilities import create_search_index, upload_documents
from pdf_ocr import process_extracted_text  # Import your processing function

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize configuration
config = Config()

# Load Azure Search credentials
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "azure_ocr"  # Use appropriate index name

# Initialize SearchIndexClient
index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

def index_on_prem_faiss(embeddings, document_ids):
    """Index documents using FAISS for on-prem setup."""
    index = faiss.IndexFlatL2(1536)
    logging.info(f"Indexing {len(embeddings)} embeddings with dimensions: {[len(e) for e in embeddings]}")
    index.add(np.array(embeddings).astype('float32'))  # Ensure embeddings are in float32 format

    # Save FAISS index to file
    faiss.write_index(index, config.FAISS_INDEX_PATH)

    # Store document mapping for future lookup
    with open(config.DOCUMENT_MAPPING_PATH, 'wb') as f:
        pickle.dump(document_ids, f)

    logging.info("FAISS index and document mapping saved.")

if __name__ == "__main__":
    # Process the PDF and generate extracted text
    pdf_path = "D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/TDP-Document.pdf"
    
    # Assuming process_extracted_text returns processed data for Azure Search or FAISS
    processed_data = process_extracted_text(pdf_path)

    if config.APPROACH == 'cloud':
        logging.info("Creating or updating Azure Search index...")
        create_search_index()

        # Create documents for upload
        documents = []
        for data in processed_data:
            content_vector = data['contentVector']
            sanitized_title = re.sub(r'[^A-Za-z0-9-_]', '_', data['title'].replace('.pdf', '').replace('.docx', '').replace('.txt', ''))

            # Validate that the content vector has the correct dimensions
            if isinstance(content_vector, list) and len(content_vector) == 1536:
                documents.append({
                    "id": sanitized_title,
                    "title": data['title'],
                    "content": data['content'],
                    "contentVector": content_vector,
                })
            else:
                logging.warning(f"Invalid contentVector or dimension mismatch for document: {data['title']}.")

        logging.info("Uploading documents to Azure Search...")
        upload_documents(documents)

    else:
        # For on-prem, handle FAISS indexing
        logging.info("Generating embeddings for FAISS indexing...")

        # Index using FAISS
        if processed_data:
            embeddings = [data['contentVector'] for data in processed_data if isinstance(data['contentVector'], list) and len(data['contentVector']) == 1536]
            document_ids = [re.sub(r'[^A-Za-z0-9-_]', '_', data['title'].replace('.pdf', '').replace('.docx', '').replace('.txt', '')) for data in processed_data if isinstance(data['contentVector'], list) and len(data['contentVector']) == 1536]

            if embeddings:
                index_on_prem_faiss(embeddings, document_ids)
            else:
                logging.error("No valid embeddings found for indexing.")

    logging.info("Indexing complete.")
