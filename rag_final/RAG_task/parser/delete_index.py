import os
import logging
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Azure Search credentials
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")  # Set your index name from the .env

# Initialize SearchIndexClient
index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

def delete_existing_index():
    """Deletes the existing Azure Search index."""
    try:
        index_client.delete_index(AZURE_SEARCH_INDEX_NAME)
        logging.info(f"Index '{AZURE_SEARCH_INDEX_NAME}' deleted.")
    except Exception as e:
        logging.error(f"Error deleting index: {e}")

if __name__ == "__main__":
    # Call the function to delete the index
    delete_existing_index()
