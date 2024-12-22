import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class Config:
    def __init__(self):
        # Select approach
        self.APPROACH = os.getenv('APPROACH', 'on-prem')  # default to 'on-prem' if not set

        # Re-ranker configuration
        self.RE_RANKER_MODEL_NAME = os.getenv('RE_RANKER_MODEL_NAME', 'cross-encoder/ms-marco-MiniLM-L-6-v2')

        # FAISS index and document mapping
        self.FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'faiss_index.index')
        self.DOCUMENT_MAPPING_PATH = os.getenv('DOCUMENT_MAPPING_PATH', 'document_mapping.pkl')

        # Embedding model and chunking
        if self.APPROACH == 'cloud':
            self.EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME_CLOUD', 'text-embedding-ada-002')
        else:
            self.EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME_ON_PREM', 'sentence-transformers/all-mpnet-base-v2')
            
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))  # Default chunk size
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))  # Default chunk overlap

        # Data folder for unstructured files
        self.DATA_FOLDER_PATH = os.getenv('DATA_FOLDER_PATH', 'D:/Genai_project/Retrieval Augmented Generation/RAG_task/data_files')

        # Cloud configuration
        if self.APPROACH == 'cloud':
            self.OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
            self.GPT_DEPLOYMENT_NAME = os.getenv('gpt_deployment_name')
            self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv('azure_openai_embedding_deployment')

            self.TENANT_ID = os.getenv('tenant_id')
            self.AZURE_SEARCH_KEY = os.getenv('AZURE_SEARCH_KEY')
            self.AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
            self.AZURE_SEARCH_ENDPOINT = os.getenv('AZURE_SEARCH_ENDPOINT')
            self.AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            self.AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME')
        else:
            # On-premises configuration
            self.OPENAI_API_BASE = None
            self.GPT_DEPLOYMENT_NAME = None
            self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT = None
            self.TENANT_ID = None
            self.AZURE_SEARCH_KEY = None
            self.AZURE_OPENAI_API_KEY = None
            self.AZURE_SEARCH_ENDPOINT = None
            self.AZURE_STORAGE_CONNECTION_STRING = None
            self.AZURE_SEARCH_INDEX_NAME = None

    def print_config(self):
        print(f"Approach: {self.APPROACH}")
        print(f"Embedding Model Name: {self.EMBEDDING_MODEL_NAME}")
        print(f"FAISS Index Path: {self.FAISS_INDEX_PATH}")
        print(f"Document Mapping Path: {self.DOCUMENT_MAPPING_PATH}")
        print(f"Re-ranker Model Name: {self.RE_RANKER_MODEL_NAME}")
        print(f"Chunk Size: {self.CHUNK_SIZE}")
        print(f"Chunk Overlap: {self.CHUNK_OVERLAP}")
        print(f"Data Folder Path: {self.DATA_FOLDER_PATH}")
        
        if self.APPROACH == 'cloud':
            print(f"Cloud API Base: {self.OPENAI_API_BASE}")
            print(f"Azure Search Endpoint: {self.AZURE_SEARCH_ENDPOINT}")
            print(f"Azure OpenAI Embedding Deployment: {self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")
            print(f"Tenant ID: {self.TENANT_ID}")
            print(f"Azure Search Key: {self.AZURE_SEARCH_KEY}")
            print(f"Azure OpenAI API Key: {self.AZURE_OPENAI_API_KEY}")
            print(f"Azure Storage Connection String: {self.AZURE_STORAGE_CONNECTION_STRING}")
            print(f"Azure Search Index Name: {self.AZURE_SEARCH_INDEX_NAME}")
        else:
            print("On-premises setup")
