import pickle

DOCUMENT_MAPPING_PATH = 'D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/parser/document_mapping.pkl'

with open(DOCUMENT_MAPPING_PATH, 'rb') as f:
    document_mapping = pickle.load(f)
    print(document_mapping)  # Check if the mapping has entries
