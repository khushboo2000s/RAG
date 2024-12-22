# import os
# import logging
# import numpy as np
# import faiss
# import pickle
# from azure.search.documents import SearchClient
# from azure.core.credentials import AzureKeyCredential
# from dotenv import load_dotenv
# from config import Config

# # Load environment variables from .env file
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Initialize configuration
# config = Config()

# # Load Azure Search credentials
# AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
# AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
# AZURE_SEARCH_INDEX_NAME = "rag-tested-data-12"  # Set your index name

# # Initialize SearchClient
# search_client = SearchClient(
#     endpoint=AZURE_SEARCH_ENDPOINT,
#     index_name=AZURE_SEARCH_INDEX_NAME,
#     credential=AzureKeyCredential(AZURE_SEARCH_KEY)
# )

# def delete_from_azure_search(doc_id):
#     """Delete a document from Azure Search by document ID."""
#     try:
#         # Perform the deletion
#         search_client.delete_documents(documents=[{"id": doc_id}])
#         logging.info(f"Document with ID '{doc_id}' deleted from Azure Search.")
#     except Exception as e:
#         logging.error(f"Error deleting document from Azure Search: {e}")

# def delete_from_faiss(doc_id):
#     """Delete a document from FAISS index based on document ID."""
#     try:
#         # Load the FAISS index
#         index = faiss.read_index(config.FAISS_INDEX_PATH)

#         # Load document mapping
#         with open(config.DOCUMENT_MAPPING_PATH, 'rb') as f:
#             document_ids = pickle.load(f)

#         # Find the index of the document ID to delete
#         if doc_id in document_ids:
#             index_id = document_ids.index(doc_id)
#             index.remove_ids(np.array([index_id]).astype('int64'))  # Remove from FAISS
#             del document_ids[index_id]  # Remove from document mapping

#             # Save the updated document mapping
#             with open(config.DOCUMENT_MAPPING_PATH, 'wb') as f:
#                 pickle.dump(document_ids, f)

#             # Save the updated FAISS index
#             faiss.write_index(index, config.FAISS_INDEX_PATH)

#             logging.info(f"Document with ID '{doc_id}' deleted from FAISS index.")
#         else:
#             logging.warning(f"Document ID '{doc_id}' not found in FAISS mapping.")
#     except Exception as e:
#         logging.error(f"Error deleting document from FAISS index: {e}")

# def delete_document(doc_id):
#     """Delete a document from both Azure Search and FAISS index."""
#     delete_from_azure_search(doc_id)
#     delete_from_faiss(doc_id)

# if __name__ == "__main__":
#     document_id_to_delete = input("Enter the document ID to delete: ")
#     delete_document(document_id_to_delete)





#############################################################
################# dlt-doc-and-chunk ########################
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

def delete_from_azure_search(doc_id):
    """Delete a document from Azure Search by document ID."""
    try:
        # Perform the deletion
        search_client.delete_documents(documents=[{"id": doc_id}])
        logging.info(f"Document with ID '{doc_id}' deleted from Azure Search.")
    except Exception as e:
        logging.error(f"Error deleting document from Azure Search: {e}")

def delete_from_faiss(doc_id):
    """Delete a document from FAISS index based on document ID."""
    try:
        # Load the FAISS index
        index = faiss.read_index(config.FAISS_INDEX_PATH)

        # Load document mapping
        with open(config.DOCUMENT_MAPPING_PATH, 'rb') as f:
            document_ids = pickle.load(f)

        # Find the index of the document ID to delete
        if doc_id in document_ids:
            index_id = document_ids.index(doc_id)
            index.remove_ids(np.array([index_id]).astype('int64'))  # Remove from FAISS
            del document_ids[index_id]  # Remove from document mapping

            # Save the updated document mapping
            with open(config.DOCUMENT_MAPPING_PATH, 'wb') as f:
                pickle.dump(document_ids, f)

            # Save the updated FAISS index
            faiss.write_index(index, config.FAISS_INDEX_PATH)

            logging.info(f"Document with ID '{doc_id}' deleted from FAISS index.")
        else:
            logging.warning(f"Document ID '{doc_id}' not found in FAISS mapping.")
    except Exception as e:
        logging.error(f"Error deleting document from FAISS index: {e}")

def delete_document(doc_id):
    """Delete a document from both Azure Search and FAISS index."""
    delete_from_azure_search(doc_id)
    delete_from_faiss(doc_id)
    logging.info(f"Deletion process completed for document ID '{doc_id}'.")

if __name__ == "__main__":
    document_id_to_delete = input("Enter the document ID to delete: ").strip()
    delete_document(document_id_to_delete)

