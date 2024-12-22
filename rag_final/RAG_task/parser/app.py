import os
import logging
from file_parser_and_chunking import RecursiveChunker, generate_embeddings, read_pdf, read_docx, read_txt  # Include file readers
from indexing_testing import index_on_prem_faiss, upload_documents, create_search_index  # Import FAISS and Azure functions
from config import Config
import re
from dotenv import load_dotenv
from langchain.schema import Document  # Ensure this is imported for Document handling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
config = Config()

# Set chunk parameters from the environment
chunk_size = int(os.getenv('CHUNK_SIZE', 1000))  # Default to 1000 characters if not specified
chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))  # Default to 200 characters overlap

def process_and_upload_documents(data_folder):
    all_embeddings = []
    document_ids = []
    documents = []

    # Iterate over files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith(('.pdf', '.docx', '.txt')):
            file_path = os.path.join(data_folder, filename)

            # Step 1: Read the file content
            if filename.endswith('.pdf'):
                text = read_pdf(file_path)
            elif filename.endswith('.docx'):
                text = read_docx(file_path)
            elif filename.endswith('.txt'):
                text = read_txt(file_path)
            else:
                logging.warning(f"Unsupported file format: {filename}")
                continue

            if not text:
                logging.warning(f"No content found in {filename}. Skipping.")
                continue

            # Step 2: Create chunks from the document
            chunker = RecursiveChunker(chunk_size, chunk_overlap)
            document = Document(page_content=text)
            chunks = chunker.chunk_documents([document])

            # Step 3: Generate embeddings for each chunk
            for i, chunk in enumerate(chunks):
                logging.info(f"Generating embeddings for chunk {i} of file {filename}...")
                embedding = generate_embeddings([chunk.page_content])  # Assuming this function returns a list of embeddings

                if embedding:
                    # Sanitize the title to create a valid document ID
                    sanitized_title = re.sub(r'[^A-Za-z0-9-_]', '_', filename.replace('.pdf', '').replace('.docx', '').replace('.txt', ''))

                    # For FAISS (on-prem)
                    all_embeddings.append(embedding[0])  # embedding[0] since generate_embeddings returns a list
                    document_ids.append(f"{sanitized_title}_{i}")
                    
                    # For Azure Search (cloud)
                    documents.append({
                        "id": f"{sanitized_title}_{i}",
                        "title": filename,
                        "content": chunk.page_content,
                        "contentVector": embedding[0]  # Assuming embedding is a list of floats
                    })

            logging.info(f"Processed {filename} with {len(chunks)} chunks.")

            # Step 4: Handle indexing/uploading based on configuration (FAISS or Azure)
            if config.APPROACH == 'cloud':
                # Upload documents to Azure Search
                if documents:
                    logging.info(f"Uploading documents for {filename} to Azure Search...")
                    create_search_index()  # Ensure the Azure index exists before uploading
                    upload_documents(documents)
                    logging.info(f"Uploaded documents for {filename} to Azure Search.")
                else:
                    logging.warning(f"No valid documents generated for {filename}.")

            else:
                # Index using FAISS (on-prem)
                if all_embeddings:
                    logging.info(f"Uploading embeddings for {filename} to FAISS.")
                    index_on_prem_faiss(all_embeddings, document_ids)
                    logging.info(f"Uploaded embeddings for {filename} to FAISS.")
                else:
                    logging.warning(f"No embeddings generated for {filename}.")
    
    logging.info("Processing and uploading completed for all documents.")

if __name__ == "__main__":
    DATA_FOLDER_PATH = os.getenv('DATA_FOLDER_PATH', "D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files")
    process_and_upload_documents(DATA_FOLDER_PATH)
