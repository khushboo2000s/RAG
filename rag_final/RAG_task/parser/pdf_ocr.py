# import os
# import json
# import logging
# from dotenv import load_dotenv
# from typing import List, Optional
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# from openai import AzureOpenAI
# from config import Config
# import re
# from search_utilities import create_search_index, upload_documents
# import easyocr
# import numpy as np
# from pdf2image import convert_from_path  # Import pdf2image

# # Setup logging
# logging.basicConfig(level=logging.INFO)

# # Load environment variables from .env file
# load_dotenv()

# # Get the data folder path and chunk settings from the .env file
# data_folder_path = os.getenv('DATA_FOLDER_PATH')
# chunk_size = int(os.getenv('CHUNK_SIZE', 1000))  # Default chunk size is 1000 characters
# chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))  # Default overlap is 200 characters

# config = Config()

# # Initialize Azure OpenAI client if in cloud approach
# if config.APPROACH == 'cloud':
#     client = AzureOpenAI(azure_endpoint=os.getenv("OPENAI_API_BASE"),
#                          api_key="bbc851a28be648d88779cd1e3de2feee",
#                          api_version='2024-02-15-preview')
# else:
#     # For on-premises, initialize the sentence transformer model
#     model_name = os.getenv('EMBEDDING_MODEL_NAME_ON_PREM', 'sentence-transformers/all-mpnet-base-v2')
#     model = SentenceTransformer(model_name)

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])  # Add other languages in the list if needed

# # Function to extract text from images using EasyOCR
# def easyocr_extract(images):
#     extracted_text = ""
#     for img in images:
#         # Convert PIL image to NumPy array
#         img_np = np.array(img)
#         # Perform OCR on the image array
#         text = reader.readtext(img_np, detail=0)  # detail=0 for only text output
#         extracted_text += " ".join(text) + "\n"
#     return extracted_text

# # Define the pdf_to_images function
# def pdf_to_images(pdf_path: str) -> List[np.ndarray]:
#     """Converts a PDF file to a list of images (as NumPy arrays)."""
#     images = convert_from_path(pdf_path)
#     return [np.array(img) for img in images]

# # Class for recursive text chunking
# class RecursiveChunker:
#     def __init__(self, chunk_size: int, chunk_overlap: int):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap

#     def chunk_documents(self, docs: List[Document]) -> List[Document]:
#         if not docs:
#             logging.warning("No documents provided for chunking.")
#             return []

#         # Check if the method exists and is available
#         try:
#             splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#                 encoding_name="cl100k_base",
#                 chunk_size=self.chunk_size,
#                 chunk_overlap=self.chunk_overlap,
#                 separators=["\n\n", "\n", " ", ""]
#             )
#         except AttributeError as e:
#             logging.error(f"Error initializing text splitter: {e}")
#             return []

#         # Split the documents into chunks
#         chunked_docs = splitter.split_documents(docs)
#         return chunked_docs

# # Function to generate embeddings
# def generate_embeddings(text_list, model="embedding"):
#     embeddings = []
#     if config.APPROACH == 'cloud':
#         logging.info('Generating embeddings using cloud model...')
#         for text in text_list:
#             try:
#                 embedding = client.embeddings.create(
#                     input=[text],
#                     model=model  # Use the model parameter as "embedding"
#                 ).data[0].embedding
#                 embeddings.append(embedding)
#             except Exception as e:
#                 logging.error(f"Error generating embedding for text: {e}")
#     else:
#         logging.info('Generating embeddings using local model...')
#         try:
#             embeddings = model.encode(text_list).tolist()  # Ensure embeddings are list
#         except Exception as e:
#             logging.error(f"Error generating embeddings using local model: {e}")
#     return embeddings

# # Function to handle extracted text, chunking, and embeddings
# def process_extracted_text(extracted_text: str):
#     # Create the search index at the beginning
#     create_search_index()

#     chunker = RecursiveChunker(chunk_size, chunk_overlap)
#     processed_documents = []  # Create a list to hold processed documents

#     logging.info("Processing extracted text...")

#     # Convert the text into a Document object
#     documents = [Document(page_content=extracted_text)]

#     # Chunk the documents
#     chunks = chunker.chunk_documents(documents)

#     logging.info(f"Number of chunks created: {len(chunks)}")

#     if not chunks:
#         logging.warning("No chunks created from the extracted text.")
#         return []

#     # Extract text from chunks
#     chunk_texts = [chunk.page_content for chunk in chunks]

#     # Generate embeddings for each chunk
#     embeddings = generate_embeddings(chunk_texts)
#     logging.info(f"Generated embeddings for {len(embeddings)} chunks.")

#     # Sanitize filename and create a document ID
#     document_id = "extracted_text_document"  # You can change this ID as needed

#     # Process each chunk and create a document structure
#     for chunk_text, embedding in zip(chunk_texts, embeddings):
#         # Validate embeddings
#         if not (isinstance(embedding, list) and all(isinstance(x, float) for x in embedding)):
#             logging.error(f"Embedding is not a list of floats for chunk.")
#             continue

#         # Create a document dictionary
#         document = {
#             "id": document_id,
#             "title": "Extracted Text Document",
#             "content": chunk_text,
#             "contentVector": embedding
#         }

#         # Append processed document to the list
#         processed_documents.append(document)

#         # Upload or index the document based on the approach
#         if config.APPROACH == 'cloud':
#             upload_documents([document])
#             logging.info("Uploaded document to Azure Search.")
#         else:
#             index_on_prem_faiss([embedding], [document_id])
#             logging.info("Uploaded embedding to FAISS.")

#     logging.info("Processing completed for the extracted text.")
    
#     return processed_documents  # Return the processed documents list


# # Main execution
# if __name__ == "__main__":
#     # Step 1: Convert PDF to images
#     pdf_path = "D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/TDP-Document.pdf"
#     images = pdf_to_images(pdf_path)

#     # Step 2: Extract text from images using EasyOCR
#     extracted_text = easyocr_extract(images)
#     print("Extracted Text:", extracted_text)  # Optional: Print the extracted text for verification

#     # Step 3: Process the extracted text for chunking and embeddings
#     processed_data = process_extracted_text(extracted_text)

#     if processed_data:
#         logging.info("Chunks and embeddings are created.")
#     else:
#         logging.error("No data was processed. Check input files or processing logic.")






import os
import json
import logging
from dotenv import load_dotenv
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from config import Config
from search_utilities import create_search_index, upload_documents
import easyocr
import numpy as np
import fitz  # PyMuPDF
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Get the data folder path and chunk settings from the .env file
data_folder_path = os.getenv('DATA_FOLDER_PATH')
chunk_size = int(os.getenv('CHUNK_SIZE', 1000))  # Default chunk size is 1000 characters
chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))  # Default overlap is 200 characters

config = Config()

# Initialize Azure OpenAI client if in cloud approach
if config.APPROACH == 'cloud':
    client = AzureOpenAI(azure_endpoint=os.getenv("OPENAI_API_BASE"),
                         api_key="bbc851a28be648d88779cd1e3de2feee",  # Use the key from environment
                         api_version='2024-02-15-preview')
else:
    # For on-premises, initialize the sentence transformer model
    model_name = os.getenv('EMBEDDING_MODEL_NAME_ON_PREM', 'sentence-transformers/all-mpnet-base-v2')
    model = SentenceTransformer(model_name)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Add other languages in the list if needed

# Define the pdf_to_images function
def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Converts a PDF file to a list of images (PIL format)."""
    images = []
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images

# Function to extract text from images using EasyOCR
def easyocr_extract(images: List[Image.Image]) -> str:
    """Extracts text from a list of images using EasyOCR."""
    extracted_text = ""
    for img in images:
        # Convert PIL Image to NumPy array
        img_np = np.array(img)
        try:
            # Perform OCR on the image
            text = reader.readtext(img_np, detail=0)  # detail=0 for only text output
            extracted_text += " ".join(text) + "\n"
        except Exception as e:
            logging.error(f"Error extracting text from image: {e}")
    return extracted_text


# Class for recursive text chunking
class RecursiveChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        if not docs:
            logging.warning("No documents provided for chunking.")
            return []

        # Initialize the text splitter
        try:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        except AttributeError as e:
            logging.error(f"Error initializing text splitter: {e}")
            return []

        # Split the documents into chunks
        chunked_docs = splitter.split_documents(docs)
        return chunked_docs

# Function to generate embeddings
def generate_embeddings(text_list, model="embedding"):
    embeddings = []
    if config.APPROACH == 'cloud':
        logging.info('Generating embeddings using cloud model...')
        for text in text_list:
            try:
                embedding = client.embeddings.create(
                    input=[text],
                    model=model  # Specify the embedding model
                ).data[0].embedding
                embeddings.append(embedding)
            except Exception as e:
                logging.error(f"Error generating embedding for text: {e}")
    else:
        logging.info('Generating embeddings using local model...')
        embeddings = model.encode(text_list).tolist()  # Ensure embeddings are list
    return embeddings

# Function to handle extracted text, chunking, and embeddings
# def process_extracted_text(extracted_text: str):
#     # Create the search index at the beginning
#     create_search_index()

#     chunker = RecursiveChunker(chunk_size, chunk_overlap)
#     processed_documents = []  # Create a list to hold processed documents

#     logging.info("Processing extracted text...")

#     # Convert the text into a Document object
#     documents = [Document(page_content=extracted_text)]

#     # Chunk the documents
#     chunks = chunker.chunk_documents(documents)

#     logging.info(f"Number of chunks created: {len(chunks)}")

#     if not chunks:
#         logging.warning("No chunks created from the extracted text.")
#         return []

#     # Extract text from chunks
#     chunk_texts = [chunk.page_content for chunk in chunks]

#     # Generate embeddings for each chunk
#     embeddings = generate_embeddings(chunk_texts)
#     logging.info(f"Generated embeddings for {len(embeddings)} chunks.")

#     # Sanitize filename and create a document ID
#     document_id = "extracted_text_document"  # You can change this ID as needed

#     # Process each chunk and create a document structure
#     for chunk_text, embedding in zip(chunk_texts, embeddings):
#         # Validate embeddings
#         if not (isinstance(embedding, list) and all(isinstance(x, float) for x in embedding)):
#             logging.error(f"Embedding is not a list of floats for chunk.")
#             continue

#         # Create a document dictionary
#         document = {
#             "id": document_id,
#             "title": "Extracted Text Document",
#             "content": chunk_text,
#             "contentVector": embedding
#         }

#         # Append processed document to the list
#         processed_documents.append(document)

#         # Upload or index the document based on the approach
#         if config.APPROACH == 'cloud':
#             upload_documents([document])
#             logging.info("Uploaded document to Azure Search.")
#         else:
#             index_on_prem_faiss([embedding], [document_id])
#             logging.info("Uploaded embedding to FAISS.")

#     logging.info("Processing completed for the extracted text.")
    
#     return processed_documents  # Return the processed documents list


# Function to handle extracted text, chunking, and embeddings
def process_extracted_text(extracted_text: str):
    # Create the search index at the beginning
    create_search_index()

    chunker = RecursiveChunker(chunk_size, chunk_overlap)
    processed_documents = []  # Create a list to hold processed documents

    logging.info("Processing extracted text...")

    # Convert the text into a Document object
    documents = [Document(page_content=extracted_text)]

    # Chunk the documents
    chunks = chunker.chunk_documents(documents)

    logging.info(f"Number of chunks created: {len(chunks)}")

    if not chunks:
        logging.warning("No chunks created from the extracted text.")
        return []

    # Extract text from chunks
    chunk_texts = [chunk.page_content for chunk in chunks]

    # Print first two or three chunks to verify
    print("First few chunks:")
    for i, chunk in enumerate(chunk_texts[:3]):
        print(f"Chunk {i + 1}: {chunk}\n")

    # Generate embeddings for each chunk
    embeddings = generate_embeddings(chunk_texts)
    logging.info(f"Generated embeddings for {len(embeddings)} chunks.")

    # Print first two or three embeddings to verify
    print("First few embeddings:")
    for i, embedding in enumerate(embeddings[:3]):
        print(f"Embedding {i + 1}: {embedding}\n")

    # Sanitize filename and create a document ID
    document_id = "extracted_text_document"  # You can change this ID as needed

    # Process each chunk and create a document structure
    for chunk_text, embedding in zip(chunk_texts, embeddings):
        # Validate embeddings
        if not (isinstance(embedding, list) and all(isinstance(x, float) for x in embedding)):
            logging.error(f"Embedding is not a list of floats for chunk.")
            continue

        # Create a document dictionary
        document = {
            "id": document_id,
            "title": "Extracted Text Document",
            "content": chunk_text,
            "contentVector": embedding
        }

        # Append processed document to the list
        processed_documents.append(document)

        # Upload or index the document based on the approach
        if config.APPROACH == 'cloud':
            upload_documents([document])
            logging.info("Uploaded document to Azure Search.")
        else:
            index_on_prem_faiss([embedding], [document_id])
            logging.info("Uploaded embedding to FAISS.")

    logging.info("Processing completed for the extracted text.")
    
    return processed_documents  # Return the processed documents list



# Main execution
if __name__ == "__main__":
    # Step 1: Convert PDF to images
    pdf_path = "D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/TDP-Document.pdf"
    images = pdf_to_images(pdf_path)

    # Step 2: Extract text from images using EasyOCR
    extracted_text = easyocr_extract(images)
    print("Extracted Text:", extracted_text)  # Optional: Print the extracted text for verification

    # Step 3: Process the extracted text for chunking and embeddings
    processed_data = process_extracted_text(extracted_text)

    if processed_data:
        logging.info("Chunks and embeddings are created.")
    else:
        logging.error("No data was processed. Check input files or processing logic.")
