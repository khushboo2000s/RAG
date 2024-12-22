import os
import logging
from dotenv import load_dotenv
from config import Config
from azure_search_ai_service.index_creation import create_search_index
from parsing_and_chunking import process_files


load_dotenv() # Load environment variables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


config = Config()

def process_documents_for_cloud_vector_store(azure_search_config, azure_openai_config):
    """
    Splits documents into chunks and stores them in the cloud-based vector database.
    """
    try:
        # Create Azure Search Index
        logging.info("Creating or updating Azure Search index...")
        get_message = create_search_index(azure_search_config)

        file_path = os.getenv('FILE_PATH', default=config.DATA_FOLDER_PATH)
        logging.info(f"Processing and chunking file: {file_path}")
        chunk_message = process_files(file_path, 1000, 200, azure_search_config, azure_openai_config) # fetch the data from different sources
        return chunk_message
    except Exception as e:
        logging.exception(f"An error occurred: {e}")


if __name__ == "__main__":
   
    azure_search_config = {
                        'endpoint': "",
                        'key': "",
                        'unstructured_rag_index': "", # add in db in tenant specific
                        }
    
    # generic
    azure_openai_config = {
                        'endpoint': "",
                        'api_key': "",
                        'api_version': "",  # Optional, you can omit this if you want to use the default
                        'azure_embeddings_deployment_name':'embedding',
                    }
    
    process_documents_for_cloud_vector_store(azure_search_config, azure_openai_config)