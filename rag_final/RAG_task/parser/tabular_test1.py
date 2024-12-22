# # import os
# # import logging
# # from dotenv import load_dotenv
# # from azure.search.documents import SearchClient
# # from azure.core.credentials import AzureKeyCredential
# # from azure.search.documents.models import VectorizedQuery
# # from openai import AzureOpenAI
# # from langchain_core.prompts import PromptTemplate
# # import urllib.parse
# # from memory import get_from_memory, update_memory  # Import memory functions

# # # Load environment variables
# # load_dotenv()

# # # Setup logging
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # # Azure Search configuration
# # AZURE_SEARCH_ENDPOINT = "https://gptkb-jcgzeo3krxxra.search.windows.net"
# # AZURE_SEARCH_KEY = "2AlK6WpXT16VPryZYgEnTohsHsnmMAvAj7Wa6Do9dCAzSeCk6q9d"
# # AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-tested-data-12')

# # # Initialize OpenAI client for Azure
# # client = AzureOpenAI(
# #     azure_endpoint="https://cog-jcgzeo3krxxra.openai.azure.com/",
# #     api_key="81505dbbd42945189028d9585b80a042",
# #     api_version="2024-02-15-preview"
# # )

# # # Define the prompt template
# # template = """Use the following pieces of context to answer the question at the end.
# # If you don't know the answer or if the question is not related to the provided context, say, "I don't know. Please ask a question related to the provided documents."
# # Always say "thanks for asking!" at the end of the answer.

# # {context}

# # Question: {question}

# # Helpful Answer:"""


# # custom_rag_prompt = PromptTemplate.from_template(template)

# # # Function to retrieve user question embedding using Azure OpenAI
# # def get_embedding(question):
# #     try:
# #         response = client.embeddings.create(
# #             model="embedding",
# #             input=[question]
# #         )
# #         return response.data[0].embedding  # Return the first embedding
# #     except Exception as e:
# #         logging.error(f"Error generating embedding: {e}")
# #         return None

# # # Function to perform index-based search in Azure
# # # Function to perform index-based search in Azure
# # def index_search(user_question, user_question_vector):
# #     search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
    
# #     # Perform index search (vector search)
# #     try:
# #         logging.info(f"User question vector: {user_question_vector}")  # Log the question vector
# #         results = search_client.search(
# #             search_text=user_question,
# #             top=5,
# #             vector_queries=[VectorizedQuery(vector=user_question_vector, k_nearest_neighbors=5, fields="contentVector")],
# #             query_type="semantic",
# #             semantic_configuration_name="my-semantic-config"
# #         )
        
# #         # Collect sources without filtering
# #         sources = [{'id': doc['id'], 'content': doc['content']} for doc in results if 'content' in doc]
# #         logging.info(f"Search results: {sources}")  # Log the results
# #         return sources if sources else None
    
# #     except Exception as e:
# #         logging.error(f"Error performing index search: {e}")
# #         return None


# # # Function to format documents for LLM, including the document ID as a clickable link
# # def format_docs(docs):
# #     if not docs:
# #         logging.warning("No documents to format.")
# #         return "No relevant documents found."

# #     formatted_sources = []
# #     print('-----formatting----')
# #     for doc in docs:
# #         if isinstance(doc, dict) and 'content' in doc and 'id' in doc:
# #             # Construct the file path using forward slashes
# #             file_name = "tdpPlan.pdf"  # Explicitly set the correct file name
# #             file_path = f"D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/{file_name}"
# #             file_link = f"file:///{urllib.parse.quote(file_path)}"  # URL encode the path with forward slashes
# #             doc_link = f"[Open Document ID: {file_name}]({file_link})"
# #             formatted_sources.append(f"{doc['content']}\n\nSource: {doc_link}")

# #     return "\n\n".join(formatted_sources)

# # # Function to perform LLM prompt with user question and retrieved sources
# # def generate_answer(context, question):
# #     # Call the OpenAI model using the prompt
# #     try:
# #         response = client.chat.completions.create(
# #             model="chat",
# #             temperature=0.1,
# #             messages=[
# #                 {"role": "system", "content": "Assistant helps users by answering questions based on provided documents."},
# #                 {"role": "user", "content": custom_rag_prompt.format(context=context, question=question)},
# #             ],
# #         )
# #         return response.choices[0].message.content
# #     except Exception as e:
# #         logging.error(f"Error generating answer from LLM: {e}")
# #         return None

# # # Main function to perform the entire workflow
# # def perform_search(user_question):
# #     logging.info(f"Performing search for: {user_question}")
    
# #     # Check if the query is already stored in memory
# #     cached_result = get_from_memory(user_question)
# #     if cached_result:
# #         logging.info("Returning result from memory.")
# #         return cached_result['answer']

# #     # Get the question embedding
# #     user_question_vector = get_embedding(user_question)
# #     if not user_question_vector:
# #         logging.warning("Failed to get question vector.")
# #         return None

# #     # Perform index search
# #     sources = index_search(user_question, user_question_vector)
# #     if not sources:
# #         logging.warning("No sources returned from index search.")
# #         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

# #     # Prepare the context and question for the LLM
# #     formatted_sources = format_docs(sources[:3])  # Format top 3 sources
# #     context = formatted_sources
# #     question = user_question

# #     # Check if the context is meaningful before generating the answer
# #     if not context or context == "No relevant documents found.":
# #         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

# #     # Generate the answer based on the sources
# #     logging.info("Generating response based on search results...")
# #     answer = generate_answer(context, question)

# #     if answer:
# #         # Create the final answer with only the answer and the source link
# #         source_link = f"Source: [Open Document ID: tdpPlan.pdf](file:///{urllib.parse.quote('D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/tdpPlan.pdf')})"
# #         full_answer = f"{answer}\n\n{source_link}"
        
# #         # Save the result to memory
# #         sources_list = ["tdpPlan.pdf"]
# #         update_memory(user_question, sources_list, full_answer)

# #         logging.info(f"Full Answer: {full_answer}")
# #         print(f"Full Answer: {full_answer}")  # Print the full answer
# #         return full_answer
# #     else:
# #         logging.warning("No answer generated by the model.")
# #         return "I'm unable to provide an answer at the moment."

# # ###############################################################################
# # if __name__ == "__main__":
# #     # Prompt user for a question
# #     query = input("Please enter your question: ")
# #     perform_search(query)






# import os
# import logging
# from dotenv import load_dotenv
# from azure.search.documents import SearchClient
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents.models import VectorizedQuery
# from openai import AzureOpenAI
# from langchain_core.prompts import PromptTemplate
# import urllib.parse
# from memory import get_from_memory, update_memory  # Import memory functions

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
#     api_key="81505dbbd42945189028d9585b80a042",
#     api_version="2024-02-15-preview"
# )

# # Define the prompt template
# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer or if the question is not related to the provided context, say, "I don't know. Please ask a question related to the provided documents."
# Always say "thanks for asking!" at the end of the answer.

# {context}

# Question: {question}

# Helpful Answer:"""


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
# # # Function to perform index-based search in Azure
# # def index_search(user_question, user_question_vector):
# #     search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
    
# #     # Perform index search (vector search)
# #     try:
# #         logging.info(f"User question vector: {user_question_vector}")  # Log the question vector
# #         results = search_client.search(
# #             search_text=user_question,
# #             top=5,
# #             vector_queries=[VectorizedQuery(vector=user_question_vector, k_nearest_neighbors=5, fields="contentVector")],
# #             query_type="semantic",
# #             semantic_configuration_name="my-semantic-config"
# #         )
        
# #         # Collect sources without filtering
# #         sources = [{'id': doc['id'], 'content': doc['content']} for doc in results if 'content' in doc]
# #         logging.info(f"Search results: {sources}")  # Log the results
# #         return sources if sources else None
    
# #     except Exception as e:
# #         logging.error(f"Error performing index search: {e}")
# #         return None



# def index_search(user_question, user_question_vector):
#     search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))

#     try:
#         logging.info(f"User question vector: {user_question_vector}")  # Log the question vector
#         # Perform index search (vector search)
#         results = search_client.search(
#             search_text=user_question,
#             top=5,
#             vector_queries=[VectorizedQuery(vector=user_question_vector, k_nearest_neighbors=5, fields="contentVector")],
#             query_type="semantic",
#             semantic_configuration_name="my-semantic-config"
#         )

#         sources = [{'id': doc['id'], 'content': doc['content']} for doc in results if 'content' in doc]
#         logging.info(f"Search results: {sources}")  # Log the search results

#         return sources if sources else None

#     except Exception as e:
#         logging.error(f"Error performing index search: {e}")
#         return None




# # Function to format documents for LLM, including the document ID as a clickable link
# def format_docs(docs):
#     if not docs:
#         logging.warning("No documents to format.")
#         return "No relevant documents found."

#     formatted_sources = []
#     print('-----formatting----')
#     for doc in docs:
#         if isinstance(doc, dict) and 'content' in doc and 'id' in doc:
#             # Construct the file path using forward slashes
#             file_name = "tdpPlan.pdf"  # Explicitly set the correct file name
#             file_path = f"D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/{file_name}"
#             file_link = f"file:///{urllib.parse.quote(file_path)}"  # URL encode the path with forward slashes
#             doc_link = f"[Open Document ID: {file_name}]({file_link})"
#             formatted_sources.append(f"{doc['content']}\n\nSource: {doc_link}")

#     return "\n\n".join(formatted_sources)

# # # Function to perform LLM prompt with user question and retrieved sources
# # def generate_answer(context, question):
# #     # Call the OpenAI model using the prompt
# #     try:
# #         response = client.chat.completions.create(
# #             model="chat",
# #             temperature=0.1,
# #             messages=[
# #                 {"role": "system", "content": "Assistant helps users by answering questions based on provided documents."},
# #                 {"role": "user", "content": custom_rag_prompt.format(context=context, question=question)},
# #             ],
# #         )
# #         return response.choices[0].message.content
# #     except Exception as e:
# #         logging.error(f"Error generating answer from LLM: {e}")
# #         return None


# def generate_answer(context, question):
#     if not context or context == "No relevant documents found.":
#         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

#     try:
#         response = client.chat.completions.create(
#             model="chat",
#             temperature=0.1,
#             messages=[
#                 {"role": "system", "content": "Assistant helps users by answering questions based on provided documents."},
#                 {"role": "user", "content": custom_rag_prompt.format(context=context, question=question)},
#             ],
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         logging.error(f"Error generating answer from LLM: {e}")
#         return None


# # # Main function to perform the entire workflow
# # def perform_search(user_question):
# #     logging.info(f"Performing search for: {user_question}")
    
# #     # Check if the query is already stored in memory
# #     cached_result = get_from_memory(user_question)
# #     if cached_result:
# #         logging.info("Returning result from memory.")
# #         return cached_result['answer']

# #     # Get the question embedding
# #     user_question_vector = get_embedding(user_question)
# #     if not user_question_vector:
# #         logging.warning("Failed to get question vector.")
# #         return None

# #     # Perform index search
# #     sources = index_search(user_question, user_question_vector)
# #     if not sources:
# #         logging.warning("No sources returned from index search.")
# #         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

# #     # Prepare the context and question for the LLM
# #     formatted_sources = format_docs(sources[:3])  # Format top 3 sources
# #     context = formatted_sources
# #     question = user_question

# #     # Check if the context is meaningful before generating the answer
# #     if not context or context == "No relevant documents found.":
# #         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

# #     # Generate the answer based on the sources
# #     logging.info("Generating response based on search results...")
# #     answer = generate_answer(context, question)

# #     if answer:
# #         # Create the final answer with only the answer and the source link
# #         source_link = f"Source: [Open Document ID: tdpPlan.pdf](file:///{urllib.parse.quote('D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/tdpPlan.pdf')})"
# #         full_answer = f"{answer}\n\n{source_link}"
        
# #         # Save the result to memory
# #         sources_list = ["tdpPlan.pdf"]
# #         update_memory(user_question, sources_list, full_answer)

# #         logging.info(f"Full Answer: {full_answer}")
# #         print(f"Full Answer: {full_answer}")  # Print the full answer
# #         return full_answer
# #     else:
# #         logging.warning("No answer generated by the model.")
# #         return "I'm unable to provide an answer at the moment."


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
#         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

#     # Prepare the context for the LLM
#     formatted_sources = format_docs(sources[:3])  # Format top 3 sources
#     context = formatted_sources

#     # Check if the context is meaningful before generating the answer
#     if not context or context == "No relevant documents found.":
#         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

#     # Generate the answer based on the sources
#     logging.info("Generating response based on search results...")
#     answer = generate_answer(context, user_question)

#     if answer:
#         full_answer = f"{answer}\n\n{context}"
#         logging.info(f"Full Answer: {full_answer}")
#         return full_answer
#     else:
#         logging.warning("No answer generated by the model.")
#         return "I'm unable to provide an answer at the moment."


# ###############################################################################
# if __name__ == "__main__":
#     # Prompt user for a question
#     query = input("Please enter your question: ")
#     perform_search(query)








# import os
# import logging
# from dotenv import load_dotenv
# from azure.search.documents import SearchClient
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents.models import VectorizedQuery
# from openai import AzureOpenAI
# from langchain_core.prompts import PromptTemplate
# import urllib.parse
# from memory import get_from_memory, update_memory  # Import memory functions

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
#     api_key="81505dbbd42945189028d9585b80a042",
#     api_version="2024-02-15-preview"
# )

# # Define the prompt template
# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer or if the question is not related to the provided context, say, "I don't know. Please ask a question related to the provided documents."
# Always say "thanks for asking!" at the end of the answer.

# {context}

# Question: {question}

# Helpful Answer:"""


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
        
#         # Collect sources without filtering
#         sources = [{'id': doc['id'], 'content': doc['content']} for doc in results if 'content' in doc]
#         logging.info(f"Search results: {sources}")  # Log the results
#         return sources if sources else None
    
#     except Exception as e:
#         logging.error(f"Error performing index search: {e}")
#         return None


# # Function to format documents for LLM, including the document ID as a clickable link
# def format_docs(docs):
#     if not docs:
#         logging.warning("No documents to format.")
#         return "No relevant documents found."

#     formatted_sources = []
#     print('-----formatting----')
#     for doc in docs:
#         if isinstance(doc, dict) and 'content' in doc and 'id' in doc:
#             # Construct the file path using forward slashes
#             file_name = "tdpPlan.pdf"  # Explicitly set the correct file name
#             file_path = f"D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/{file_name}"
#             file_link = f"file:///{urllib.parse.quote(file_path)}"  # URL encode the path with forward slashes
#             doc_link = f"[Open Document ID: {file_name}]({file_link})"
#             formatted_sources.append(f"{doc['content']}\n\nSource: {doc_link}")

#     return "\n\n".join(formatted_sources)

# # Function to perform LLM prompt with user question and retrieved sources
# def generate_answer(context, question):
#     # Call the OpenAI model using the prompt
#     try:
#         response = client.chat.completions.create(
#             model="chat",
#             temperature=0.1,
#             messages=[
#                 {"role": "system", "content": "Assistant helps users by answering questions based on provided documents."},
#                 {"role": "user", "content": custom_rag_prompt.format(context=context, question=question)},
#             ],
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         logging.error(f"Error generating answer from LLM: {e}")
#         return None

# # Main function to perform the entire workflow
# def perform_search(user_question):
#     logging.info(f"Performing search for: {user_question}")
    
#     # Check if the query is already stored in memory
#     cached_result = get_from_memory(user_question)
#     if cached_result:
#         logging.info("Returning result from memory.")
#         return cached_result['answer']

#     # Get the question embedding
#     user_question_vector = get_embedding(user_question)
#     if not user_question_vector:
#         logging.warning("Failed to get question vector.")
#         return None

#     # Perform index search
#     sources = index_search(user_question, user_question_vector)
#     if not sources:
#         logging.warning("No sources returned from index search.")
#         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

#     # Prepare the context and question for the LLM
#     formatted_sources = format_docs(sources[:3])  # Format top 3 sources
#     context = formatted_sources
#     question = user_question

#     # Check if the context is meaningful before generating the answer
#     if not context or context == "No relevant documents found.":
#         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

#     # Generate the answer based on the sources
#     logging.info("Generating response based on search results...")
#     answer = generate_answer(context, question)

#     if answer:
#         # Create the final answer with only the answer and the source link
#         source_link = f"Source: [Open Document ID: tdpPlan.pdf](file:///{urllib.parse.quote('D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/tdpPlan.pdf')})"
#         full_answer = f"{answer}\n\n{source_link}"
        
#         # Save the result to memory
#         sources_list = ["tdpPlan.pdf"]
#         update_memory(user_question, sources_list, full_answer)

#         logging.info(f"Full Answer: {full_answer}")
#         print(f"Full Answer: {full_answer}")  # Print the full answer
#         return full_answer
#     else:
#         logging.warning("No answer generated by the model.")
#         return "I'm unable to provide an answer at the moment."

# ###############################################################################
# if __name__ == "__main__":
#     # Prompt user for a question
#     query = input("Please enter your question: ")
#     perform_search(query)





##########################################################################
########Testing....#######################################################
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

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Azure Search configuration
AZURE_SEARCH_ENDPOINT = "https://gptkb-jcgzeo3krxxra.search.windows.net"
AZURE_SEARCH_KEY = "2AlK6WpXT16VPryZYgEnTohsHsnmMAvAj7Wa6Do9dCAzSeCk6q9d"
AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-1')

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
# def index_search(user_question, user_question_vector):
#     search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))

#     try:
#         logging.info(f"User question vector: {user_question_vector}")  # Log the question vector
#         # Perform index search (vector search)
#         results = search_client.search(
#             search_text=user_question,
#             top=5,
#             vector_queries=[VectorizedQuery(vector=user_question_vector, k_nearest_neighbors=5, fields="contentVector")],
#             query_type="semantic",
#             semantic_configuration_name="my-semantic-config"
#         )

#         sources = [{'id': doc['id'], 'content': doc['content']} for doc in results if 'content' in doc]
#         logging.info(f"Search results: {sources}")  # Log the search results

#         return sources if sources else None

#     except Exception as e:
#         logging.error(f"Error performing index search: {e}")
#         return None


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
        
#         # Check if results are empty or irrelevant (assumed condition for "out of box")
#         if not results or all('content' not in doc for doc in results):
#             logging.warning("No relevant documents found in index search.")
#             return None  # Return None if no relevant documents are found
        
#         # Collect sources if relevant documents are found
#         sources = [{'id': doc['id'], 'content': doc['content']} for doc in results if 'content' in doc]
#         logging.info(f"Search results: {sources}")  # Log the results
#         return sources  # Return sources if found
    
#     except Exception as e:
#         logging.error(f"Error performing index search: {e}")
#         return None


################Testing...###########################

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
        
        # Collect sources if relevant documents are found
        sources = [{'id': doc['id'], 'content': doc['content']} for doc in results if 'content' in doc]
        
        # Implement content relevance check here
        relevant_sources = [
            source for source in sources 
            if any(keyword in source['content'].lower() for keyword in user_question.lower().split())
        ]

        # Log relevant sources if found
        if relevant_sources:
            logging.info(f"Relevant search results: {relevant_sources}")  # Log only the relevant results
        else:
            logging.info("No relevant documents found in index search.")  # Log when no relevant documents are found
        
        return relevant_sources  # Return relevant sources if found
    
    except Exception as e:
        logging.error(f"Error performing index search: {e}")
        return []  # Return an empty list in case of an error

#################END...##########################################



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
            file_name = "tdpPlan.pdf"  # Explicitly set the correct file name
            file_path = f"D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/{file_name}"
            file_link = f"file:///{urllib.parse.quote(file_path)}"  # URL encode the path with forward slashes
            doc_link = f"[Open Document ID: {file_name}]({file_link})"
            formatted_sources.append(f"{doc['content']}\n\nSource: {doc_link}")

    return "\n\n".join(formatted_sources)

# Function to perform LLM prompt with user question and retrieved sources
def generate_answer(context, question):
    if not context or context == "No relevant documents found.":
        return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

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
#         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

#     # Prepare the context for the LLM
#     formatted_sources = format_docs(sources[:3])  # Format top 3 sources
#     context = formatted_sources

#     # Check if the context is meaningful before generating the answer
#     if not context or context == "No relevant documents found.":
#         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

#     # Generate the answer based on the sources
#     logging.info("Generating response based on search results...")
#     answer = generate_answer(context, user_question)

#     if answer:
#         full_answer = f"{answer}\n\n{context}"
#         logging.info(f"Full Answer: {full_answer}")
#         return full_answer
#     else:
#         logging.warning("No answer generated by the model.")
#         return "I'm unable to provide an answer at the moment."



# def perform_search(user_question):
#     logging.info(f"Performing search for: {user_question}")
    
#     # Check if the query is already stored in memory
#     cached_result = get_from_memory(user_question)
#     if cached_result:
#         logging.info("Returning result from memory.")
#         return cached_result['answer']

#     # Get the question embedding
#     user_question_vector = get_embedding(user_question)
#     if not user_question_vector:
#         logging.warning("Failed to get question vector.")
#         return None

#     # Perform index search
#     sources = index_search(user_question, user_question_vector)
    
#     # Check if sources is None (indicating no documents found)
#     if sources is None:
#         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

#     # Prepare the context and question for the LLM
#     formatted_sources = format_docs(sources[:3])  # Format top 3 sources
#     context = formatted_sources
#     question = user_question

#     # Check if the context is meaningful before generating the answer
#     if not context or context == "No relevant documents found.":
#         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

#     # Generate the answer based on the sources
#     logging.info("Generating response based on search results...")
#     answer = generate_answer(context, question)

#     if answer:
#         # Create the final answer with only the answer and the source link
#         source_link = f"Source: [Open Document ID: tdpPlan.pdf](file:///{urllib.parse.quote('D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/tdpPlan.pdf')})"
#         full_answer = f"{answer}\n\n{source_link}"
        
#         # Save the result to memory
#         sources_list = ["tdpPlan.pdf"]
#         update_memory(user_question, sources_list, full_answer)

#         logging.info(f"Full Answer: {full_answer}")
#         print(f"Full Answer: {full_answer}")  # Print the full answer
#         return full_answer
#     else:
#         logging.warning("No answer generated by the model.")
#         return "I'm unable to provide an answer at the moment."







def perform_search(user_question):
    logging.info(f"Performing search for: {user_question}")

    # Check if the query is already stored in memory
    cached_result = get_from_memory(user_question)
    if cached_result:
        logging.info("Returning result from memory.")
        return cached_result['answer']

    # Get the question embedding
    user_question_vector = get_embedding(user_question)
    if not user_question_vector:
        logging.warning("Failed to get question vector.")
        return None

    # Perform index search
    sources = index_search(user_question, user_question_vector)

    # Check if the sources are empty
    if not sources:
        logging.info("No relevant sources found from index search.")  # Log when no relevant sources are found
        return "I don't know. Please ask a question related to the provided documents. Thanks for asking!"

    # Log search results only if relevant sources are found
    # logging.info(f"Search results found: {sources}")

    # Prepare the context and question for the LLM
    formatted_sources = format_docs(sources[:3])  # Format top 3 sources
    context = formatted_sources
    question = user_question

    # Generate the answer based on the sources
    logging.info("Generating response based on search results...")
    answer = generate_answer(context, question)

    if answer:
        # Create the final answer with only the answer and the source link
        source_link = f"Source: [Open Document ID: tdpPlan.pdf](file:///{urllib.parse.quote('D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/tdpPlan.pdf')})"
        full_answer = f"{answer}\n\n{source_link}"

        # Save the result to memory
        sources_list = ["tdpPlan.pdf"]
        update_memory(user_question, sources_list, full_answer)

        logging.info(f"Full Answer: {full_answer}")
        print(f"Full Answer: {full_answer}")  # Print the full answer
        return full_answer
    else:
        logging.warning("No answer generated by the model.")
        return "I'm unable to provide an answer at the moment."




###############################################################################
if __name__ == "__main__":
    # Prompt user for a question
    # query = input("Please enter your question: ")
    query = " What is deque in DS?"
    perform_search(query)








