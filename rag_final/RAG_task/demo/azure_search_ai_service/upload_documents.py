from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import logging


load_dotenv() # Load environment variables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def upload_documents(documents, azure_search_config):
    """
    Function to Upload documents to Azure Search Index.
    """
    message=''
    try:
        search_client = SearchClient(
                                endpoint=azure_search_config['endpoint'],
                                index_name=azure_search_config['unstructured_rag_index'],
                                credential=AzureKeyCredential(azure_search_config['key'])
                            )
        result = search_client.upload_documents(documents)
        logging.info(f"Documents uploaded successfully: {result}")
        message = "Documents uploaded successfully."  # Success message
        
    except Exception as e:
        logging.error(f"Error uploading documents: {e}")
        message = "An error occurred while uploading documents. Please try again later."  # Custom error message

    return message  # Return the message
        
        