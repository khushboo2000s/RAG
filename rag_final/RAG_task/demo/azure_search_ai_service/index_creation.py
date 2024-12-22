import os
import logging
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
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
    VectorSearchAlgorithmKind,
    HnswParameters,
    SearchField
)

from azure.core.credentials import AzureKeyCredential

load_dotenv() # Load environment variables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def create_search_index(azure_search_config):
    """
    Create or update Azure Search Index.
    """
    try:
        index_name = azure_search_config.get('unstructured_rag_index')

        client = SearchIndexClient(
            endpoint = azure_search_config['endpoint'],
            credential=AzureKeyCredential(azure_search_config['key'])
        )

        # Define the index schema
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, retrievable=True, stored=True,
                        vector_search_dimensions=1536, vector_search_profile_name="my-vector-config"),
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
        result = client.create_or_update_index(index)
        logging.info("Index created or updated. Result: %s",result)
        message= "Index created and updated succesfully."
        return message

    except Exception as e:
        logging.error(f"An error occurred while creating or updating the index: {e}")
        message = "Error in creating or updating Index in Azure"
        return message
        