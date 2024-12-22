# import os
# import logging
# import numpy as np
# from azure.search.documents.indexes import SearchIndexClient
# from azure.search.documents import SearchClient
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents.indexes.models import (
#     SearchIndex,
#     SimpleField,
#     SearchFieldDataType,
#     VectorSearch,
#     HnswAlgorithmConfiguration,
#     VectorSearchProfile,
#     SemanticConfiguration,
#     SemanticPrioritizedFields,
#     SemanticField,
#     SemanticSearch,
#     SearchableField,
#     SearchField
# )
# from config1 import Config
# from dotenv import load_dotenv
# import faiss
# import pickle
# import re
# from file_parser_and_chunking1 import process_files
# from search_utilities1 import create_search_index, upload_documents

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

# # Initialize SearchIndexClient
# index_client = SearchIndexClient(
#     endpoint=AZURE_SEARCH_ENDPOINT,
#     credential=AzureKeyCredential(AZURE_SEARCH_KEY)
# )


# def index_on_prem_faiss(embeddings, document_ids):
#     """Index documents using FAISS for on-prem setup."""
#     index = faiss.IndexFlatL2(1536)
#     index.add(np.array(embeddings).astype('float32'))  # Ensure embeddings are in float32 format

#     # Save FAISS index to file
#     faiss.write_index(index, config.FAISS_INDEX_PATH)

#     # Store document mapping for future lookup
#     with open(config.DOCUMENT_MAPPING_PATH, 'wb') as f:
#         pickle.dump(document_ids, f)

#     logging.info("FAISS index and document mapping saved.")

# if __name__ == "__main__":
#     data_folder_path = os.getenv('DATA_FOLDER_PATH')
#     chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
#     chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))

#     # Process files
#     processed_data = process_files(data_folder_path, chunk_size, chunk_overlap)

#     if config.APPROACH == 'cloud':
#         logging.info("Creating or updating Azure Search index...")
#         create_search_index()

#         # Create documents for upload
#         documents = []
#         for idx, data in enumerate(processed_data):
#             content_vector = data['contentVector']


#             # Sanitize the title to create a valid document ID
#             sanitized_title = re.sub(r'[^A-Za-z0-9-_]', '_', data['title'].replace('.pdf', '').replace('.docx', '').replace('.txt', ''))    #
    
            
#             # Validate that the content vector has the correct dimensions
#             if isinstance(content_vector, list) and len(content_vector) == 1536:
#                 print("Content_vector:", content_vector)
#                 documents.append({
#                     "id": f"{sanitized_title}_{idx}",
#                     "title": data['title'],
#                     "content": data['content'],
#                     "contentVector": content_vector,  # Directly use the valid content vector
#                 })
#             else:
#                 logging.warning(f"Document {idx}: Invalid contentVector or dimension mismatch.")

#         logging.info("Uploading documents to Azure Search...")
#         upload_documents(documents)

#     else:
#         # For on-prem, handle FAISS indexing
#         logging.info("Generating embeddings for FAISS indexing...")

#         # Index using FAISS
#         if processed_data:
#             embeddings = [data['contentVector'] for data in processed_data if isinstance(data['contentVector'], list) and len(data['contentVector']) == 1536]
#             # Use sanitized title for document IDs
#             document_ids = [f"{sanitized_title}_{idx}" for idx, data in enumerate(processed_data) if isinstance(data['contentVector'], list) and len(data['contentVector']) == 1536]

            
#             if embeddings:
#                 index_on_prem_faiss(embeddings, document_ids)
#             else:
#                 logging.error("No valid embeddings found for indexing.")

#     logging.info("Indexing complete.")







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
from config1 import Config
from dotenv import load_dotenv
import faiss
import pickle
import re
from testing import process_files
from search_utilities1 import create_search_index, upload_documents

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize configuration
config = Config()

# Load Azure Search credentials
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "rag-tested-data-12"  # Set your index name

# Initialize SearchIndexClient
index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

def sanitize_title(title):
    """Sanitize the title to create a valid document ID."""
    return re.sub(r'[^A-Za-z0-9-_]', '_', title)

def index_on_prem_faiss(embeddings, document_ids):
    """Index documents using FAISS for on-prem setup."""
    index = faiss.IndexFlatL2(1536)
    index.add(np.array(embeddings).astype('float32'))  # Ensure embeddings are in float32 format

    # Save FAISS index to file
    faiss.write_index(index, config.FAISS_INDEX_PATH)

    # Store document mapping for future lookup
    with open(config.DOCUMENT_MAPPING_PATH, 'wb') as f:
        pickle.dump(document_ids, f)

    logging.info("FAISS index and document mapping saved.")

if __name__ == "__main__":
    data_folder_path = os.getenv('DATA_FOLDER_PATH')
    chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))

    # Process files
    logging.info(f"Processing files from: {data_folder_path}")
    processed_data = process_files(data_folder_path, chunk_size, chunk_overlap)

    logging.info(f"Processed data: {processed_data}")

    if config.APPROACH == 'cloud':
        logging.info("Creating or updating Azure Search index...")
        create_search_index()

        # Create documents for upload
        documents = []
        for idx, data in enumerate(processed_data):
            content_vector = data['contentVector']
            sanitized_title = sanitize_title(data['title'].replace('.pdf', '').replace('.docx', '').replace('.txt', ''))

            # Validate that the content vector has the correct dimensions
            if isinstance(content_vector, list) and len(content_vector) == 1536:
                documents.append({
                    "id": f"{sanitized_title}_{idx}",
                    "title": data['title'],
                    "content": data['content'],
                    "contentVector": content_vector,  # Directly use the valid content vector
                })
            else:
                logging.warning(f"Document {idx}: Invalid contentVector or dimension mismatch.")

        logging.info(f"Documents prepared for upload: {documents}")

        if documents:  # Only attempt upload if documents exist
            logging.info("Uploading documents to Azure Search...")
            upload_documents(documents)
        else:
            logging.warning("No documents to upload.")

    else:
        # For on-prem, handle FAISS indexing
        logging.info("Generating embeddings for FAISS indexing...")

        # Index using FAISS
        if processed_data:
            embeddings = [data['contentVector'] for data in processed_data if isinstance(data['contentVector'], list) and len(data['contentVector']) == 1536]
            # Use sanitized title for document IDs
            document_ids = [f"{sanitize_title(data['title'])}_{idx}" for idx, data in enumerate(processed_data) if isinstance(data['contentVector'], list) and len(data['contentVector']) == 1536]

            if embeddings:
                index_on_prem_faiss(embeddings, document_ids)
            else:
                logging.error("No valid embeddings found for indexing.")

    logging.info("Indexing complete.")
