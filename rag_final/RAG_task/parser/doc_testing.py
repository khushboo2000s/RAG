## .....Deleted document is present or not.... ##

import os
import logging
import numpy as np
import faiss
import pickle
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from config import Config

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize configuration
config = Config()

# Load Azure Search credentials
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "rag-1"  # Set your index name

# Initialize SearchClient
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

def check_in_azure_search(doc_filename):
    """Check if a document exists in Azure Search by filename."""
    try:
        # Perform a search query using the document filename
        results = search_client.search(search_text=doc_filename)
        
        found = False  # Flag to track if the document is found

        # Iterate over results to check document fields
        for result in results:
            logging.info(f"Azure Search result: {result}")  # Print all fields of the document
            
            # Check if the filename exists in the document's fields
            if doc_filename in result.get("file_name", ""):
                logging.info(f"Document with filename '{doc_filename}' is present in Azure Search.")
                found = True
                break
        
        if not found:
            logging.info(f"Document with filename '{doc_filename}' is NOT present in Azure Search.")
        return found
    except Exception as e:
        logging.error(f"Error checking document in Azure Search: {e}")
        return False


def check_in_faiss(doc_filename):
    """Check if a document exists in FAISS index based on filename."""
    try:
        # Load the FAISS index
        index = faiss.read_index(config.FAISS_INDEX_PATH)

        # Load document mapping
        with open(config.DOCUMENT_MAPPING_PATH, 'rb') as f:
            document_filenames = pickle.load(f)

        # Check if the document filename is in the mapping
        if doc_filename in document_filenames:
            logging.info(f"Document with filename '{doc_filename}' is present in FAISS index.")
            return True
        else:
            logging.info(f"Document with filename '{doc_filename}' is NOT present in FAISS index.")
            return False
    except Exception as e:
        logging.error(f"Error checking document in FAISS index: {e}")
        return False

def check_document_present(doc_filename):
    """Check if a document is present in both Azure Search and FAISS index."""
    is_in_azure = check_in_azure_search(doc_filename)
    is_in_faiss = check_in_faiss(doc_filename)

    if is_in_azure or is_in_faiss:
        logging.info(f"Document with filename '{doc_filename}' is still present.")
    else:
        logging.info(f"Document with filename '{doc_filename}' is no longer present in both systems.")

if __name__ == "__main__":
    document_filename_to_check = input("Enter the document filename to check: ").strip()
    check_document_present(document_filename_to_check)
