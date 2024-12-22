from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Azure Search configuration (could also move to config.py)
AZURE_SEARCH_ENDPOINT = os.getenv('AZURE_SEARCH_ENDPOINT')
AZURE_SEARCH_KEY = os.getenv('AZURE_SEARCH_KEY')
AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME')

def azure_index_search(user_question, user_question_vector):
    """
    Performs an index-based search using Azure Search.
    """
    print(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX_NAME)
    search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
    try:
        logging.info(f"User question vector: {user_question_vector}")  
        results = search_client.search(
            search_text=user_question,
            top=5,
            vector_queries=[VectorizedQuery(vector=user_question_vector, k_nearest_neighbors=5, fields="contentVector")],
            query_type="semantic",
            semantic_configuration_name="my-semantic-config"
        )
        return [{'id': doc['id'], 'content': doc['content']} for doc in results if 'content' in doc]
    except Exception as e:
        logging.error(f"Error during Azure index search: {e}")
        return None
