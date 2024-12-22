import os
import logging
import json
import numpy as np
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchIndex,
    HnswParameters,
    SimpleField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchProfile
)
from config import Config
from dotenv import load_dotenv
import faiss
import pickle
from file_parser_and_chunking import process_files

from azure.search.documents.indexes import SearchIndexClient
from sentence_transformers import SentenceTransformer
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
    HnswParameters,
    SearchField
)
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
 
 


# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize configuration
config = Config()

# Load Azure Search credentials
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "rag-tested-data-"  # Set your index name

# Initialize SearchIndexClient
index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# Create or update search index with vector search
# def create_search_index():
#     try:
#         existing_index = index_client.get_index(AZURE_SEARCH_INDEX_NAME)
#         logging.info(f"Index '{AZURE_SEARCH_INDEX_NAME}' already exists. Updating...")

#         # Check if 'contentVector' field exists and update if necessary
#         if "contentVector" not in [field.name for field in existing_index.fields]:
#             existing_index.fields.append(
#                 SimpleField(
#                     name="contentVector",
#                     type=SearchFieldDataType.Collection(SearchFieldDataType.Single),  # Correct type for embeddings
#                     searchable=False,
#                     filterable=False,
#                     facetable=False,
#                     sortable=False,
#                     retrievable=True,
#                     stored=True,
#                 )
#             )
#             index_client.create_or_update_index(existing_index)
#             logging.info(f"Index '{AZURE_SEARCH_INDEX_NAME}' updated to include 'contentVector'.")

#     except Exception as e:
#         if "not found" in str(e):
#             logging.info(f"Index '{AZURE_SEARCH_INDEX_NAME}' does not exist. Creating new index...")
#             index = SearchIndex(
#                 name=AZURE_SEARCH_INDEX_NAME,
#                 fields=[
#                     SimpleField(name="id", type=SearchFieldDataType.String, key=True),
#                     SimpleField(name="title", type=SearchFieldDataType.String, searchable=True),
#                     SimpleField(name="content", type=SearchFieldDataType.String, searchable=True),
#                     SimpleField(
#                         name="contentVector",
#                         type=SearchFieldDataType.Collection(SearchFieldDataType.Single),  # Correct type for embeddings
#                         searchable=True,
#                         filterable=False,
#                         facetable=False,
#                         sortable=False,
#                         retrievable=True,
#                         stored=True,
#                     ),
#                 ],
#                 vector_search=VectorSearch(
#                     algorithms=[HnswAlgorithmConfiguration(
#                         name="hnsw_config",
#                         kind=VectorSearchAlgorithmKind.HNSW,
#                         parameters=HnswParameters(metric="cosine"),
#                     )],
#                     profiles=[VectorSearchProfile(name="embedding_profile", algorithm_configuration_name="hnsw_config")]
#                 )
#             )
#             index_client.create_index(index)
#             logging.info(f"Index '{AZURE_SEARCH_INDEX_NAME}' created.")
#         else:
#             logging.error(f"Error creating or updating the index: {e}")
###############################################################################
### Testing......
def create_search_index():
    """
    Create or update Azure Search Index.
    """
    index_name = "rag-tested-data-four"  # Replace with tenant name
 
    client = SearchIndexClient(
        endpoint=config.AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(config.AZURE_SEARCH_KEY)
    )
 
    # Define the index schema
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        # Ensure the contentVector is set as a collection of floats (numeric vectors)
        SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=1536, vector_search_profile_name="my-vector-config"),
        ]
    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-algorithms-config")],
        algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")],
    )
   
 
    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="content")]
        )
    )
 
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(configurations=[semantic_config])
    )
 
    # Create or update the index
    try:
        result = client.create_or_update_index(index)
        logging.info("Index created or updated")
    except Exception as e:
        logging.error(f"An error occurred while creating or updating the index: {e}")



# Function to validate and flatten embeddings
def validate_and_flatten_embeddings(processed_data):
    valid_embeddings = []
    document_ids = []

    for idx, data in enumerate(processed_data):
        content_vector = data['contentVector']
        if isinstance(content_vector, list) and all(isinstance(i, (int, float)) for i in content_vector):
            if len(content_vector) == 1536:
                logging.info(f"Document {idx}: contentVector is valid and has dimension = {len(content_vector)}.")
                valid_embeddings.append(np.array(content_vector).flatten().tolist())
                document_ids.append(data['id'])
            else:
                logging.warning(f"Document {idx}: Embedding dimension does not match expected 1536!")
        else:
            logging.error(f"Document {idx}: contentVector is not valid: {content_vector}")

    return valid_embeddings, document_ids

# Function to upload documents to Azure Search
def upload_documents(documents):
    print(documents)
    search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
    try:
        results = search_client.upload_documents(documents)
        logging.info(f"Uploaded {len(documents)} documents.")
        return results
    except Exception as e:
        logging.error(f"Error uploading documents: {e}")
        return None

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
    processed_data = process_files(data_folder_path, chunk_size, chunk_overlap)

    if config.APPROACH == 'cloud':
        logging.info("Creating or updating Azure Search index...")
        create_search_index()

        # Validate and flatten embeddings
        embeddings, document_ids = validate_and_flatten_embeddings(processed_data)

        # Create documents for upload
        documents = []
        for idx, data in enumerate(processed_data):
            content_vector = data['contentVector']
            
            # Validate that the content vector has the correct dimensions
            if isinstance(content_vector, list) and len(content_vector) == 1536:
                documents.append({
                    "id": f"{data['title'].replace('.pdf', '').replace(' ', '_')}_{idx}",
                    "title": data['title'],
                    "content": data['content'],
                    "contentVector": content_vector,  # Directly use the valid content vector
                })
            else:
                logging.warning(f"Document {idx}: Invalid contentVector or dimension mismatch.")

        logging.info("Uploading documents to Azure Search...")
        upload_documents(documents)

    else:
        # For on-prem, handle FAISS indexing
        logging.info("Generating embeddings for FAISS indexing...")

        # Validate and flatten embeddings
        embeddings, document_ids = validate_and_flatten_embeddings(processed_data)

        # Index using FAISS
        if embeddings:
            index_on_prem_faiss(embeddings, document_ids)
        else:
            logging.error("No valid embeddings found for indexing.")

    logging.info("Indexing complete.")
