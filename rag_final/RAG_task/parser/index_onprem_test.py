# #### on - prem - index ##########

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
# from config import Config
# from dotenv import load_dotenv
# import faiss
# import pickle
# import re
# # from file_parser_and_chunking import process_files
# from file_parser_onprem import process_files     ##
# from search_utilities_onprem import create_search_index, upload_documents

# # Load environment variables from .env file
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Initialize configuration
# config = Config()

# # Load Azure Search credentials
# AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
# AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
# AZURE_SEARCH_INDEX_NAME = "rag-1"  # Set your index name

# # Initialize SearchIndexClient
# index_client = SearchIndexClient(
#     endpoint=AZURE_SEARCH_ENDPOINT,
#     credential=AzureKeyCredential(AZURE_SEARCH_KEY)
# )


# # def index_on_prem_faiss(embeddings, document_ids):
# #     """Index documents using FAISS for on-prem setup."""
# #     index = faiss.IndexFlatL2(768)
# #     logging.info(f"Indexing {len(embeddings)} embeddings with dimensions: {[len(e) for e in embeddings]}")   #
# #     index.add(np.array(embeddings).astype('float32'))  # Ensure embeddings are in float32 format

# #     # Save FAISS index to file
# #     faiss.write_index(index, config.FAISS_INDEX_PATH)

# #     # Store document mapping for future lookup
# #     with open(config.DOCUMENT_MAPPING_PATH, 'wb') as f:
# #         pickle.dump(document_ids, f)

# #     logging.info("FAISS index and document mapping saved.")

# ################# Changes - add #########################################
# # Function to create the FAISS index for on-premises use
# def create_faiss_index(dimensions):
#     logging.info("Creating FAISS index...")
#     index = faiss.IndexFlatL2(dimensions)  # Create a flat L2 index
#     logging.info(f"FAISS index created with dimensions: {dimensions}.")
#     return index


# #####################################################################
# # def index_on_prem_faiss(embeddings, document_ids, processed_data):
# #     """Index documents using FAISS for on-prem setup."""
# #     index = faiss.IndexFlatL2(1536)   #768
# #     logging.info(f"Indexing {len(embeddings)} embeddings with dimensions: {[len(e) for e in embeddings]}")
# #     index.add(np.array(embeddings).astype('float32'))  # Ensure embeddings are in float32 format

# #     # Store document mapping as a dictionary    ### chnages #####
# #     # document_mapping = {idx: {"content": processed_data[idx]['content'], "title": processed_data[idx]['title']} for idx in range(len(document_ids))}
# #     document_mapping = {doc_id: {"title": data['title']} for doc_id, data in zip(document_ids, processed_data)}


# #     # Save FAISS index to file
# #     faiss.write_index(index, config.FAISS_INDEX_PATH)

# #     ##### Changes -add ######
# #     # document_mapping = {doc_id: {"title": title} for doc_id, title in zip(document_ids, processed_documents)}

# #     # Store document mapping for future lookup
# #     with open(config.DOCUMENT_MAPPING_PATH, 'wb') as f:
# #         pickle.dump(document_mapping, f)

# #     logging.info("FAISS index and document mapping saved.")

# #######################################################################
# ######################################################################
# # Function to upload documents to the FAISS index
# def index_on_prem_faiss(embeddings, document_ids, document_mapping):
#     try:
#         logging.info(f"Indexing {len(embeddings)} embeddings to FAISS.")
#         index = create_faiss_index(dimensions=768)  # Ensure dimensions match your embedding model

#         # Before and after adding embeddings to FAISS
#         logging.info(f"Indexing {len(embeddings)} embeddings.")
#         logging.info(f"Embeddings shape: {np.array(embeddings).shape}")


#         index.add(np.array(embeddings).astype('float32'))  # Ensure the embeddings are float32
#         logging.info(f"Successfully indexed {len(embeddings)} embeddings.")

#         # Save index to disk for later retrieval
#         faiss.write_index(index, config.FAISS_INDEX_PATH)
#         logging.info("FAISS index saved to disk.")

#         # Save document mapping for future lookup
#         with open(config.DOCUMENT_MAPPING_PATH, 'wb') as f:
#             pickle.dump(document_mapping, f)
#         logging.info("Document mapping saved.")
        
#     except Exception as e:
#         logging.error(f"Error indexing documents to FAISS: {e}")

# ###############################################################################
# ###############################################################################



# if __name__ == "__main__":
#     data_folder_path = os.getenv('DATA_FOLDER_PATH')
#     chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
#     chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))

#     # Process files
#     processed_data = process_files(data_folder_path, chunk_size, chunk_overlap)

#     # Initialize lists for documents and document IDs
#     documents = []
#     document_ids = []
#     document_mapping = {}


#     if config.APPROACH == 'cloud':
#         logging.info("Creating or updating Azure Search index...")
#         create_search_index()

#         # Create documents for upload
#         # documents = []
#         for idx, data in enumerate(processed_data):
#             content_vector = data['contentVector']


#             # Sanitize the title to create a valid document ID
#             sanitized_title = re.sub(r'[^A-Za-z0-9-_]', '_', data['title'].replace('.pdf', '').replace('.docx', '').replace('.txt', ''))    #
    
            
#             # Validate that the content vector has the correct dimensions
#             if isinstance(content_vector, list) and len(content_vector) == 768:   #1536
#                 print("Content_vector:", content_vector)
#                 documents.append({
#                     "id": f"{sanitized_title}_{idx}",
#                     "title": data['title'],
#                     "content": data['content'],
#                     "contentVector": content_vector,  # Directly use the valid content vector
#                 })
#                 document_ids.append(f"{sanitized_title}_{idx}")  #
#             else:
#                 logging.warning(f"Document {idx}: Invalid contentVector or dimension mismatch.")

#         logging.info("Uploading documents to Azure Search...")
#         upload_documents(documents)

#     else:
#         # For on-prem, handle FAISS indexing
#         logging.info("Generating embeddings for FAISS indexing...")

# #         # Index using FAISS
# #         if processed_data:
# #             embeddings = [data['contentVector'] for data in processed_data if isinstance(data['contentVector'], list) and len(data['contentVector']) == 768]  #1536
# #             # Use sanitized title for document IDs
# #             # document_ids = [f"{sanitized_title}_{idx}" for idx, data in enumerate(processed_data) if isinstance(data['contentVector'], list) and len(data['contentVector']) == 768]

# # ########### Changes -add ##################################

# #             for idx, data in enumerate(processed_data):
# #                 sanitized_title = re.sub(r'[^A-Za-z0-9-_]', '_', data['title'].replace('.pdf', '').replace('.docx', '').replace('.txt', ''))
# #                 document_ids.append(f"{sanitized_title}_{idx}")




            
# #             if embeddings:
# #                 index_on_prem_faiss(embeddings, document_ids)
# #                 logging.info("FAISS indexing complete.")
# #             else:
# #                 logging.error("No valid embeddings found for indexing.")

# #     logging.info("Indexing complete.")



#         # Index using FAISS
#         if processed_data:
#             embeddings = []
#             for idx, data in enumerate(processed_data):
#                 sanitized_title = re.sub(r'[^A-Za-z0-9-_]', '_', data['title'].replace('.pdf', '').replace('.docx', '').replace('.txt', ''))
#                 document_ids.append(f"{sanitized_title}_{idx}")

#                 if isinstance(data['contentVector'], list) and len(data['contentVector']) == 768:
#                     embeddings.append(data['contentVector'])
#                     document_mapping[f"{sanitized_title}_{idx}"] = {"title": data['title']}  # Create document mapping

#             # Log document mapping creation
#             logging.info(f"Document mapping created with keys: {list(document_mapping.keys())}")  # <-- Add this line here


#             if embeddings:
#                 index_on_prem_faiss(embeddings, document_ids, document_mapping)
#                 logging.info("FAISS indexing complete.")
#             else:
#                 logging.error("No valid embeddings found for indexing.")

#     logging.info("Indexing complete.")





######### new-document-mapping #######################
import os
import logging
import numpy as np
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
import faiss
import pickle
import re
from file_parser_onprem import process_files
from search_utilities_onprem import create_search_index, upload_documents
from config import Config
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()



# Set up logging configuration
log_file_path = 'application.log'  # Path to log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),  # Log file handler
        logging.StreamHandler()  # Console handler
    ]
)

# Create logger
logging = logging.getLogger(__name__)

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize configuration
config = Config()

# Load Azure Search credentials
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "rag-1"  # Set your index name

# Initialize SearchIndexClient
index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# Function to create the FAISS index for on-premises use
def create_faiss_index(dimensions):
    logging.info("Creating FAISS index...")
    index = faiss.IndexFlatL2(dimensions)  # Create a flat L2 index
    logging.info(f"FAISS index created with dimensions: {dimensions}.")
    return index

# Function to upload documents to the FAISS index
def index_on_prem_faiss(embeddings, document_ids, document_mapping):
    try:
        logging.info(f"Indexing {len(embeddings)} embeddings to FAISS.")
        index = create_faiss_index(dimensions=768)  # Ensure dimensions match your embedding model

        # Before and after adding embeddings to FAISS
        logging.info(f"Indexing {len(embeddings)} embeddings.")
        logging.info(f"Embeddings shape: {np.array(embeddings).shape}")

        index.add(np.array(embeddings).astype('float32'))  # Ensure the embeddings are float32
        logging.info(f"Successfully indexed {len(embeddings)} embeddings.")

        # Save index to disk for later retrieval
        faiss.write_index(index, config.FAISS_INDEX_PATH)
        logging.info("FAISS index saved to disk.")

        # Save document mapping for future lookup
        with open(config.DOCUMENT_MAPPING_PATH, 'wb') as f:
            pickle.dump(document_mapping, f)
        logging.info("Document mapping saved.")


        with open(config.DOCUMENT_MAPPING_PATH.replace('.pkl', '.json'), 'w') as file:  #
            json.dump(document_mapping, file)   #
        logging.info(f"Document mapping written to {config.DOCUMENT_MAPPING_PATH.replace('.pkl', '.json')}")   #

        
    except Exception as e:
        logging.error(f"Error indexing documents to FAISS: {e}")

if __name__ == "__main__":
    data_folder_path = os.getenv('DATA_FOLDER_PATH')
    chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))

    # Process files
    processed_data = process_files(data_folder_path, chunk_size, chunk_overlap)

    # Initialize lists for documents and document IDs
    documents = []
    document_ids = []
    document_mapping = {}

    if config.APPROACH == 'cloud':
        logging.info("Creating or updating Azure Search index...")
        create_search_index()

        # Create documents for upload
        for idx, data in enumerate(processed_data):
            content_vector = data['contentVector']

            # Sanitize the title to create a valid document ID
            sanitized_title = re.sub(r'[^A-Za-z0-9-_]', '_', data['title'].replace('.pdf', '').replace('.docx', '').replace('.txt', '').replace('.csv', '').replace('.xlsx', '').replace('.xls', ''))

            # Validate that the content vector has the correct dimensions
            if isinstance(content_vector, list) and len(content_vector) == 768:
                logging.info(f"Valid content vector for document {idx}: {content_vector}")
                documents.append({
                    # "id": f"{sanitized_title}_{idx}",
                    "id" : f"{sanitized_title}_{i}_{filename}",
                    "title": data['title'],
                    "content": data['content'],
                    "contentVector": content_vector,  # Directly use the valid content vector
                })
                # document_ids.append(f"{sanitized_title}_{idx}")
                document_ids.append(f"{sanitized_title}_{i}_{filename}")
            else:
                logging.warning(f"Document {idx}: Invalid contentVector or dimension mismatch.")

        logging.info("Uploading documents to Azure Search...")
        upload_documents(documents)

    else:
        # For on-prem, handle FAISS indexing
        logging.info("Generating embeddings for FAISS indexing...")

        if processed_data:
            embeddings = []
            for idx, data in enumerate(processed_data):
                sanitized_title = re.sub(r'[^A-Za-z0-9-_]', '_', data['title'].replace('.pdf', '').replace('.docx', '').replace('.txt', '').replace('.csv', '').replace('.xlsx', '').replace('.xls', ''))
                # document_ids.append(f"{sanitized_title}_{idx}")
                document_ids.append(f"{sanitized_title}_{i}_{filename}")

                if isinstance(data['contentVector'], list) and len(data['contentVector']) == 768:
                    embeddings.append(data['contentVector'])
                    # document_mapping[f"{sanitized_title}_{idx}"] = {"title": data['title'], "content": data['content']}  # Create complete document mapping
                    document_mapping[f"{sanitized_title}_{i}_{filename}"] = {"title": data['title'], "content": data['content']}

            # Log document mapping creation
            logging.info(f"Document mapping created with keys: {list(document_mapping.keys())}")

            if embeddings:
                index_on_prem_faiss(embeddings, document_ids, document_mapping)
                logging.info("FAISS indexing complete.")
            else:
                logging.error("No valid embeddings found for indexing.")

    logging.info("Indexing complete.")
