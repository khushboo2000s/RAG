import os
import logging
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from langchain_core.prompts import PromptTemplate
import urllib.parse

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Azure Search configuration
AZURE_SEARCH_ENDPOINT = "https://gptkb-jcgzeo3krxxra.search.windows.net"
AZURE_SEARCH_KEY = ""
AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-tested-data-12')

# Initialize OpenAI client for Azure
client = AzureOpenAI(
    azure_endpoint="",
    api_key="",
    api_version="2024-02-15-preview"
)

# Define the prompt template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""


########################################################################



custom_rag_prompt = PromptTemplate.from_template(template)

# Function to retrieve user question embedding using Azure OpenAI
def get_embedding(question):
    try:
        response = client.embeddings.create(
            model="embedding",
            input=[question]
        )
        return response.data[0].embedding  # Return the first embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None

# Function to perform index-based search in Azure
def index_search(user_question, user_question_vector):
    search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
    
    # Perform index search (vector search)
    try:
        logging.info(f"User question vector: {user_question_vector}")  # Log the question vector
        results = search_client.search(
            search_text=user_question,
            top=5,
            vector_queries=[VectorizedQuery(vector=user_question_vector, k_nearest_neighbors=5, fields="contentVector")],
            query_type="semantic",
            semantic_configuration_name="my-semantic-config"
        )
        
        # Collect sources
        sources = [{'id': doc['id'], 'content': doc['content']} for doc in results if 'content' in doc]
        logging.info(f"Search results: {sources}")  # Log the results
        return sources
    
    except Exception as e:
        logging.error(f"Error performing index search: {e}")
        return None

# Function to format documents for LLM, including the document ID as a clickable link
def format_docs(docs):
    if not docs:
        logging.warning("No documents to format.")
        return "No relevant documents found."

    formatted_sources = []
    print('-----formatting----')
    for doc in docs:
        if isinstance(doc, dict) and 'content' in doc and 'id' in doc:
            # Construct the file path using forward slashes
            file_name = "2022ESG Report.pdf"  # Explicitly set the correct file name
            file_path = f"D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/{file_name}"
            file_link = f"file:///{urllib.parse.quote(file_path)}"  # URL encode the path with forward slashes
            doc_link = f"[Open Document ID: {file_name}]({file_link})"
            formatted_sources.append(f"{doc['content']}\n\nSource: {doc_link}")

    return "\n\n".join(formatted_sources)

# Function to perform LLM prompt with user question and retrieved sources
def generate_answer(context, question):
    # Call the OpenAI model using the prompt
    try:
        response = client.chat.completions.create(
            model="chat",
            temperature=0.1,
            messages=[
                {"role": "system", "content": "Assistant helps users by answering questions based on provided documents."},
                {"role": "user", "content": custom_rag_prompt.format(context=context, question=question)},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating answer from LLM: {e}")
        return None

# Main function to perform the entire workflow
def perform_search(user_question):
    logging.info(f"Performing search for: {user_question}")
    
    # Get the question embedding
    user_question_vector = get_embedding(user_question)
    if not user_question_vector:
        logging.warning("Failed to get question vector.")
        return None

    # Perform index search
    sources = index_search(user_question, user_question_vector)
    if not sources:
        logging.warning("No sources returned from index search.")
        return None

    # Prepare the context and question for the LLM
    formatted_sources = format_docs(sources[:3])  # Format top 3 sources
    context = formatted_sources
    question = user_question

    # Generate the answer based on the sources
    logging.info("Generating response based on search results...")
    answer = generate_answer(context, question)
    # if answer:
    #     # Create the final answer with the desired format
    #     full_answer = f"{answer}\n\n{formatted_sources}"
    #     logging.info(f"Full Answer: {full_answer}")
    #     return full_answer
    # else:
    #     logging.warning("No answer generated by the model.")


    if answer:
        # Create the final answer with only the answer and the source link
        # Assuming formatted_sources contains the links you need
        source_link = f"Source: [Open Document ID: 2022ESG Report.pdf](file:///{urllib.parse.quote('D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/2022ESG Report.pdf')})"
        full_answer = f"{answer}\n\n{source_link}"
        logging.info(f"Full Answer: {full_answer}")
        return full_answer
    else:
        logging.warning("No answer generated by the model.")
###############################################################################
if __name__ == "__main__":
    # Example user question
    query =  "What is the consensus revenue for Squarespace?"

    perform_search(query)
