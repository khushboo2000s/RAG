import os
import logging
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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Azure Search credentials
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "rag-1"

# Initialize SearchIndexClient
index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

def create_search_index():
    """
    Create or update Azure Search Index.
    """
    index_name = "rag-1"

    client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    # Define the index schema
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, retrievable=True, stored=True, vector_search_dimensions=1536, vector_search_profile_name="my-vector-config"),
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
        logging.info("Index created or updated. Result: %s", result)

        index = client.get_index(index_name)
        logging.info("Index retrieved: %s", index)
        print("Index:", index)

    except Exception as e:
        logging.error(f"An error occurred while creating or updating the index: {e}")
        return None

# def upload_documents(documents):
#     search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
#     try:
#         results = search_client.upload_documents(documents)
#         logging.info(f"Uploaded {len(documents)} documents.")
#         return results
#     except Exception as e:
#         logging.error(f"Error uploading documents: {e}")
#         return None




def upload_documents(documents):
    search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
    try:
        # Log information for each document being uploaded
        for document in documents:
            logging.info(f"Preparing to upload document: {document['id']}, Content Length: {len(document['content'])}")
        
        results = search_client.upload_documents(documents)
        logging.info(f"Uploaded {len(documents)} documents.")
        return results
    except Exception as e:
        logging.error(f"Error uploading documents: {e}")
        return None
