import os
import logging
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from langchain_core.prompts import PromptTemplate
import urllib.parse
from memory import get_from_memory, update_memory  # Import memory functions
import numpy as np

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Azure Search configuration
AZURE_SEARCH_ENDPOINT = "https://gptkb-jcgzeo3krxxra.search.windows.net"
AZURE_SEARCH_KEY = "2AlK6WpXT16VPryZYgEnTohsHsnmMAvAj7Wa6Do9dCAzSeCk6q9d"
AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-tested-data-12')

# Initialize OpenAI client for Azure
client = AzureOpenAI(
    azure_endpoint="https://cog-jcgzeo3krxxra.openai.azure.com/",
    api_key="81505dbbd42945189028d9585b80a042",
    api_version="2024-02-15-preview"
)

# Define the prompt template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer or if the question is not related to the provided context, say, "I don't know. Please ask a question related to the provided documents."
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

# Cosine similarity function for vector comparison
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

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

# # Function to perform index-based search in Azure
# def index_search(user_question, user_question_vector, similarity_threshold=0.7):
#     search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
    
#     try:
#         logging.info(f"User question vector: {user_question_vector}")  # Log the question vector
#         results = search_client.search(
#             search_text=user_question,
#             top=5,
#             vector_queries=[VectorizedQuery(vector=user_question_vector, k_nearest_neighbors=5, fields="contentVector")],
#             query_type="semantic",
#             semantic_configuration_name="my-semantic-config"
#         )

#         sources = []
#         keywords = user_question.lower().split()  # Split the question into keywords

#         for doc in results:
#             content = doc.get('content', '')
#             content_vector = doc.get('contentVector', None)

#             similarity = cosine_similarity(user_question_vector, content_vector)
#             if similarity >= similarity_threshold and any(keyword in content.lower() for keyword in keywords):
#                 sources.append({'id': doc['id'], 'content': content})

#         logging.info(f"Filtered search results: {sources}")
#         return sources if sources else None

#     except Exception as e:
#         logging.error(f"Error performing index search: {e}")
#         return None


def index_search(user_question, user_question_vector, similarity_threshold=0.7):
    search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
    
    try:
        logging.info(f"User question vector: {user_question_vector}")  # Log the question vector
        
        # Perform index search (vector search)
        results = search_client.search(
            search_text=user_question,
            top=5,
            vector_queries=[VectorizedQuery(vector=user_question_vector, k_nearest_neighbors=5, fields="contentVector")],
            query_type="semantic",
            semantic_configuration_name="my-semantic-config"
        )
        
        # Collect sources if they meet the similarity threshold
        sources = []
        for doc in results:
            content = doc.get('content', '')
            content_vector = doc.get('contentVector', None)
            
            # Ensure that both vectors are present before calculating similarity
            if content_vector is None:
                logging.warning(f"Document {doc['id']} has no content vector. Skipping this document.")
                continue
            
            similarity = cosine_similarity(user_question_vector, content_vector)
            
            # Include documents that meet the similarity threshold
            if similarity >= similarity_threshold:
                sources.append({'id': doc['id'], 'content': content})
        
        logging.info(f"Filtered search results: {sources}")  # Log the filtered results
        return sources if sources else None

    except Exception as e:
        logging.error(f"Error performing index search: {e}")
        return None



# Function to format documents for LLM, including the document ID as a clickable link
def format_docs(docs):
    if not docs:
        logging.warning("No documents to format.")
        return "No relevant documents found."

    formatted_sources = []
    for doc in docs:
        if 'content' in doc and 'id' in doc:
            file_name = "tdpPlan.pdf"  # Explicitly set the correct file name
            file_path = f"D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/{file_name}"
            file_link = f"file:///{urllib.parse.quote(file_path)}"  # URL encode the path
            doc_link = f"[Open Document ID: {file_name}]({file_link})"
            formatted_sources.append(f"{doc['content']}\n\nSource: {doc_link}")

    return "\n\n".join(formatted_sources)

# Function to perform LLM prompt with user question and retrieved sources
def generate_answer(context, question):
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

# # Main function to perform the entire workflow
# def perform_search(user_question):
#     logging.info(f"Performing search for: {user_question}")
    
#     cached_result = get_from_memory(user_question)
#     if cached_result:
#         logging.info("Returning result from memory.")
#         return cached_result['answer']

#     user_question_vector = get_embedding(user_question)
#     if not user_question_vector:
#         logging.warning("Failed to get question vector.")
#         return "I couldn't process your question. Please try again."

#     sources = index_search(user_question, user_question_vector)
#     if not sources:
#         logging.warning("No relevant documents returned from vector search.")
#         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

#     formatted_sources = format_docs(sources[:3])
#     context = formatted_sources

#     if not context or context == "No relevant documents found.":
#         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

#     answer = generate_answer(context, user_question)

#     if answer:
#         source_link = f"Source: [Open Document ID: tdpPlan.pdf](file:///{urllib.parse.quote('D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/tdpPlan.pdf')})"
#         full_answer = f"{answer}\n\n{source_link}"

#         sources_list = ["tdpPlan.pdf"]
#         update_memory(user_question, sources_list, full_answer)

#         logging.info(f"Full Answer: {full_answer}")
#         return full_answer
#     else:
#         logging.warning("No answer generated by the model.")
#         return "I'm unable to provide an answer at the moment."



def perform_search(user_question):
    logging.info(f"Performing search for: {user_question}")

    # Get the question embedding
    user_question_vector = get_embedding(user_question)
    if not user_question_vector:
        logging.warning("Failed to get question vector.")
        return "I couldn't process your question. Please try again."

    # Perform index search
    sources = index_search(user_question, user_question_vector)
    if not sources:
        logging.warning("No relevant documents returned from vector search.")
        return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

    # Prepare the context for the LLM (if LLM is part of the flow)
    formatted_sources = format_docs(sources[:3])  # Format top 3 sources
    context = formatted_sources

    # Check if the context is meaningful before generating the answer
    if not context or context == "No relevant documents found.":
        return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

    # If you're using an LLM here:
    answer = generate_answer(context, user_question)

    if answer:
        full_answer = f"{answer}\n\n{context}"
        logging.info(f"Full Answer: {full_answer}")
        return full_answer
    else:
        logging.warning("No answer generated by the model.")
        return "I'm unable to provide an answer at the moment."



###############################################################################
if __name__ == "__main__":
    query = input("Please enter your question: ")
    perform_search(query)
