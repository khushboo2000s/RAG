from dotenv import load_dotenv
from openai import AzureOpenAI
from config import Config
import logging
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_embeddings(text_list, azure_openai_config):
    """
    Generate embeddings for the given list of texts.
    """
    embeddings = []
    try:
        # Initialize Azure OpenAI client using the provided config
        client = AzureOpenAI(
            azure_endpoint = azure_openai_config['endpoint'],
            api_key = azure_openai_config['api_key'],
            api_version = azure_openai_config.get('api_version', '2024-02-15-preview')  # Default if not provided
            )
        
         # Use the provided deployment name from config
        model_name = azure_openai_config.get('azure_embeddings_deployment_name')
        logging.info('Generating embeddings using cloud model...')
        embeddings = [client.embeddings.create(input=[text], model=model_name).data[0].embedding for text in text_list]
        
    except Exception as e:
        logging.exception(f"Error generating embeddings: {e}")
    return embeddings