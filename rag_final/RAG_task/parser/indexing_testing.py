###########CLOUD#########################################################
#######################################################################


import os
import json
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
# from file_parser_and_chunking import process_files
from excel_1 import process_files
from search_utilities import create_search_index, upload_documents

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize configuration
config = Config()

# Load Azure Search credentials
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "excel_rag"  # Set your index name   #rag-1

# Initialize SearchIndexClient
index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)


###########################################
# Function to store the documents in a separate JSON file
def save_documents_to_file(documents, file_path='D:\\Genai_project\\Retrieval Augmented Generation\\rag_final\\RAG_task\\file_testing\\final_upload.json'):
    with open(file_path, 'w') as f:
        json.dump(documents, f, indent=4)
    logging.info(f"Documents have been saved to {file_path}")

###############################################

def index_on_prem_faiss(embeddings, document_ids):
    """Index documents using FAISS for on-prem setup."""
    index = faiss.IndexFlatL2(1536)
    logging.info(f"Indexing {len(embeddings)} embeddings with dimensions: {[len(e) for e in embeddings]}")   #
    index.add(np.array(embeddings).astype('float32'))  # Ensure embeddings are in float32 format

    # Save FAISS index to file
    faiss.write_index(index, config.FAISS_INDEX_PATH)

    # Store document mapping for future lookup
    with open(config.DOCUMENT_MAPPING_PATH, 'wb') as f:
        pickle.dump(document_ids, f)

    logging.info("FAISS index and document mapping saved.")

if __name__ == "__main__":
    data_folder_path = os.getenv('DATA_FOLDER_PATH')
    chunk_size = int(os.getenv('CHUNK_SIZE', 1500))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 300))

    # Process files
    processed_data = process_files(data_folder_path, chunk_size, chunk_overlap)

    if config.APPROACH == 'cloud':
        logging.info("Creating or updating Azure Search index...")
        create_search_index()

    #     # Create documents for upload
    #     documents = []
    #     for idx, data in enumerate(processed_data):
    #         content_vector = data['contentVector']


    #         # Sanitize the title to create a valid document ID
    #         sanitized_title = re.sub(r'[^A-Za-z0-9-_]', '_', data['title'].replace('.pdf', '').replace('.docx', '').replace('.txt', ''))    #
    
            
    #         # Validate that the content vector has the correct dimensions
    #         if isinstance(content_vector, list) and len(content_vector) == 1536:
    #             print("Content_vector:", content_vector)
    #             documents.append({
    #                 "id": f"{sanitized_title}_{idx}",
    #                 "title": data['title'],
    #                 "content": data['content'],
    #                 "contentVector": content_vector,  # Directly use the valid content vector
    #             })
    #         else:
    #             logging.warning(f"Document {idx}: Invalid contentVector or dimension mismatch.")

    #     logging.info("Uploading documents to Azure Search...")
    #     upload_documents(documents)

    # else:
    #     # For on-prem, handle FAISS indexing
    #     logging.info("Generating embeddings for FAISS indexing...")

    #     # Index using FAISS
    #     if processed_data:
    #         embeddings = [data['contentVector'] for data in processed_data if isinstance(data['contentVector'], list) and len(data['contentVector']) == 1536]
    #         # Use sanitized title for document IDs
    #         document_ids = [f"{sanitized_title}_{idx}" for idx, data in enumerate(processed_data) if isinstance(data['contentVector'], list) and len(data['contentVector']) == 1536]

            
    #         if embeddings:
    #             index_on_prem_faiss(embeddings, document_ids)
    #         else:
    #             logging.error("No valid embeddings found for indexing.")

    # logging.info("Indexing complete.")
###################################document_id#####################################
        # create documents for upload....

        # Create documents for upload
        documents = []
        # for data in processed_data:
        for idx, data in enumerate(processed_data):  #
            content_vector = data['contentVector']

            # Sanitize the title to create a valid document ID
            sanitized_title = re.sub(r'[^A-Za-z0-9-_]', '_', data['title'].replace('.pdf', '').replace('.docx', '').replace('.txt', ''))

            # Generate a unique document ID for each chunk by appending the chunk index
            document_id = f"{sanitized_title}_chunk_{idx}"  # Add chunk index for uniqueness


            # Validate that the content vector has the correct dimensions
            if isinstance(content_vector, list) and len(content_vector) == 1536:
                print("Content_vector:", content_vector)
                documents.append({
                    "id": document_id,  # Use the same document ID for all chunks
                    "title": data['title'],
                    "content": data['content'],
                    "contentVector": content_vector,  # Directly use the valid content vector
                })
            else:
                logging.warning(f"Invalid contentVector or dimension mismatch for document: {data['title']}.")

#####################
        # Save the documents to a JSON file before uploading
        save_documents_to_file(documents)  #
##########################

        logging.info("Uploading documents to Azure Search...")
        upload_documents(documents)

    else:
        # For on-prem, handle FAISS indexing
        logging.info("Generating embeddings for FAISS indexing...")

        # Index using FAISS
        if processed_data:
            embeddings = [data['contentVector'] for data in processed_data if isinstance(data['contentVector'], list) and len(data['contentVector']) == 1536]
            # Use sanitized title for document IDs
            document_ids = [re.sub(r'[^A-Za-z0-9-_]', '_', data['title'].replace('.pdf', '').replace('.docx', '').replace('.txt', '')) for data in processed_data if isinstance(data['contentVector'], list) and len(data['contentVector']) == 1536]

            if embeddings:
                index_on_prem_faiss(embeddings, document_ids)
            else:
                logging.error("No valid embeddings found for indexing.")

    logging.info("Indexing complete.")



##########################################################################
################ ON-PREM #################################################

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
# from file_parser_and_chunking import process_files
# from search_utilities import create_search_index, upload_documents

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

# #####################################################################
# def index_on_prem_faiss(embeddings, document_ids, processed_data):
#     """Index documents using FAISS for on-prem setup."""
#     index = faiss.IndexFlatL2(768)
#     logging.info(f"Indexing {len(embeddings)} embeddings with dimensions: {[len(e) for e in embeddings]}")
#     index.add(np.array(embeddings).astype('float32'))  # Ensure embeddings are in float32 format

#     # Store document mapping as a dictionary
#     document_mapping = {idx: {"content": processed_data[idx]['content'], "title": processed_data[idx]['title']} for idx in range(len(document_ids))}

#     # Save FAISS index to file
#     faiss.write_index(index, config.FAISS_INDEX_PATH)

#     # Store document mapping for future lookup
#     with open(config.DOCUMENT_MAPPING_PATH, 'wb') as f:
#         pickle.dump(document_mapping, f)

#     logging.info("FAISS index and document mapping saved.")

# #######################################################################




# if __name__ == "__main__":
#     data_folder_path = os.getenv('DATA_FOLDER_PATH')
#     chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
#     chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))

#     # Process files
#     processed_data = process_files(data_folder_path, chunk_size, chunk_overlap)

#     # Initialize lists for documents and document IDs
#     documents = []
#     document_ids = []


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
#             if isinstance(content_vector, list) and len(content_vector) == 768:
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

#         # Index using FAISS
#         if processed_data:
#             embeddings = [data['contentVector'] for data in processed_data if isinstance(data['contentVector'], list) and len(data['contentVector']) == 768]
#             # Use sanitized title for document IDs
#             # document_ids = [f"{sanitized_title}_{idx}" for idx, data in enumerate(processed_data) if isinstance(data['contentVector'], list) and len(data['contentVector']) == 768]

            
#             if embeddings:
#                 index_on_prem_faiss(embeddings, document_ids, processed_data)
#             else:
#                 logging.error("No valid embeddings found for indexing.")

#     logging.info("Indexing complete.")






