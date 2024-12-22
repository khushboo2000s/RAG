from openai import AzureOpenAI
import logging
import os

# Initialize OpenAI client for Azure
azure_openai_client = AzureOpenAI(
    azure_endpoint="https://cog-jcgzeo3krxxra.openai.azure.com/",
    api_key="81505dbbd42945189028d9585b80a042",  # Ensure to use the correct API key from the environment
    api_version="2024-02-15-preview"
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