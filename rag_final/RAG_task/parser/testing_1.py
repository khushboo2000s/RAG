# import os
# import logging
# from dotenv import load_dotenv
# from azure.search.documents import SearchClient
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents.models import VectorizedQuery
# from openai import AzureOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # Load environment variables
# load_dotenv()

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Azure Search configuration
# AZURE_SEARCH_ENDPOINT = "https://gptkb-jcgzeo3krxxra.search.windows.net"
# AZURE_SEARCH_KEY = "2AlK6WpXT16VPryZYgEnTohsHsnmMAvAj7Wa6Do9dCAzSeCk6q9d"
# AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-tested-data-12')

# # Initialize OpenAI client for Azure
# client = AzureOpenAI(
#     azure_endpoint="https://cog-jcgzeo3krxxra.openai.azure.com/",
#     api_key="81505dbbd42945189028d9585b80a042",  # Ensure to use the correct API key from the environment
#     api_version="2024-02-15-preview"
# )

# # Define the prompt template
# # template = """Use the following pieces of context to answer the question at the end.
# # If you don't know the answer, just say that you don't know; don't try to make up an answer.
# # Use three sentences maximum and keep the answer as concise as possible.
# # Always say "thanks for asking!" at the end of the answer.

# # {context}

# # Question: {question}

# # Helpful Answer:"""

# template = """
# Answer the question based only on the following context:
# {context}
#  - -
# Answer the question based on the above context: {question}
# """

# custom_rag_prompt = PromptTemplate.from_template(template)

# # Function to retrieve user question embedding using Azure OpenAI
# def get_embedding(question):
#     try:
#         response = client.embeddings.create(
#             model="embedding",
#             input=[question]
#         )
#         return response.data[0].embedding  # Return the first embedding
#     except Exception as e:
#         logging.error(f"Error generating embedding: {e}")
#         return None

# # Function to perform index-based search in Azure
# def index_search(user_question, user_question_vector):
#     search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
    
#     # Perform index search (vector search)
#     try:
#         logging.info(f"User question vector: {user_question_vector}")  # Log the question vector
#         results = search_client.search(
#             search_text=user_question,
#             top=5,
#             vector_queries=[VectorizedQuery(vector=user_question_vector, k_nearest_neighbors=5, fields="contentVector")],
#             query_type="semantic",
#             semantic_configuration_name="my-semantic-config"
#         )
        
#         # Collect sources
#         sources = [{'id': doc['id'], 'content': doc['content']} for doc in results if 'content' in doc]
#         logging.info(f"Search results: {sources}")  # Log the results
#         return sources
    
#     except Exception as e:
#         logging.error(f"Error performing index search: {e}")
#         return None

# # Function to format documents for LLM
# def format_docs(docs):
#     if not docs:
#         logging.warning("No documents to format.")
#         return "No relevant documents found."
    
#     return "\n\n".join(doc['content'] for doc in docs if isinstance(doc, dict) and 'content' in doc)

# # Function to perform LLM prompt with user question and retrieved sources
# def generate_answer(user_question, sources):
#     limited_sources = format_docs(sources[:3])  # Format top 3 sources
    
#     # Prepare the input for the prompt
#     USER_MESSAGE = {"context": limited_sources, "question": user_question}
    
#     # Call the OpenAI model using the prompt
#     try:
#         response = client.chat.completions.create(
#             model="chat",
#             temperature=0.7,
#             messages=[
#                 {"role": "system", "content": "Assistant helps users by answering questions based on provided documents."},
#                 {"role": "user", "content": custom_rag_prompt.format(**USER_MESSAGE)},
#             ],
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         logging.error(f"Error generating answer from LLM: {e}")
#         return None

# # Main function to perform the entire workflow
# def perform_search(user_question):
#     logging.info(f"Performing search for: {user_question}")
    
#     # Get the question embedding
#     user_question_vector = get_embedding(user_question)
#     if not user_question_vector:
#         logging.warning("Failed to get question vector.")
#         return None

#     # Perform index search
#     sources = index_search(user_question, user_question_vector)
#     if not sources:
#         logging.warning("No sources returned from index search.")
#         return None

#     # Generate the answer based on the sources
#     logging.info("Generating response based on search results...")
#     answer = generate_answer(user_question, sources)
#     if answer:
#         logging.info(f"Answer: {answer}")
#     else:
#         logging.warning("No answer generated by the model.")

# if __name__ == "__main__":
#     # Example user question
#     query = "give me all the information about hamas"
#     perform_search(query)






import os
import logging
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough

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
    api_key="81505dbbd42945189028d9585b80a042",  # Ensure to use the correct API key from the environment
    api_version="2024-02-15-preview"
)

# Define the prompt template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""


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

# Function to format documents for LLM
def format_docs(docs):
    if not docs:
        logging.warning("No documents to format.")
        return "No relevant documents found."
    
    return "\n\n".join(doc['content'] for doc in docs if isinstance(doc, dict) and 'content' in doc)

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

    # Prepare the context and question for rag_chain
    formatted_sources = format_docs(sources[:3])  # Format top 3 sources
    context = formatted_sources
    question = user_question

    # Generate the answer based on the sources
    logging.info("Generating response based on search results...")
    answer = generate_answer(context, question)
    if answer:
        logging.info(f"Answer: {answer}")
    else:
        logging.warning("No answer generated by the model.")

if __name__ == "__main__":
    # Example user question
    query = " How Iran Raises and Moves Funds in Support of Terrorism"
    perform_search(query)
