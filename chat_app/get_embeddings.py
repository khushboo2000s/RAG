from openai import AzureOpenAI
import logging
import os

# Initialize OpenAI client for Azure
azure_openai_client = AzureOpenAI(
    azure_endpoint="",
    api_key="",  # Ensure to use the correct API key from the environment
    api_version=""
)

def get_embedding_from_openai(question):
    """
    Generate embeddings using Azure OpenAI.
    """
    try:
        response = azure_openai_client.embeddings.create(model="embedding", input=[question])
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None