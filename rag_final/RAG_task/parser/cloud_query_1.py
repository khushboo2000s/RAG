# import os
# import logging
# from dotenv import load_dotenv
# from azure.search.documents import SearchClient
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents.models import VectorizedQuery
# from openai import AzureOpenAI
# from langchain_core.prompts import PromptTemplate
# import urllib.parse
# from memory import get_from_memory, update_memory, get_chat_history  # Import for chat history
# from langchain_community.chat_message_histories import ChatMessageHistory  # Import for chat history

# # Load environment variables
# load_dotenv()

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Azure Search configuration
# AZURE_SEARCH_ENDPOINT = "https://gptkb-jcgzeo3krxxra.search.windows.net"
# AZURE_SEARCH_KEY = os.getenv('AZURE_SEARCH_KEY')  # Store sensitive data securely
# AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-1')

# # Initialize OpenAI client for Azure
# client = AzureOpenAI(
#     azure_endpoint="https://cog-jcgzeo3krxxra.openai.azure.com/",
#     api_key="81505dbbd42945189028d9585b80a042",
#     api_version="2024-02-15-preview"
# )

# # # Define the prompt template
# # template = """Use the following pieces of context to answer the question at the end.
# # If you don't know the answer or if the question is not related to the provided context, say, "I don't know. Please ask a question related to the provided documents."
# # Always say "thanks for asking!" at the end of the answer.

# # {context}

# # Question: {question}

# # Helpful Answer:"""

# ################### Template - Testing ####################################
# # Revised prompt template -(Conversational and Engaging)
# template = """Let’s find the answer to your question! Here’s what I gathered from the context. If you don’t see a relevant answer or if it’s unrelated, please say: "I don't know. Please ask a question related to the provided documents."
# Remember to always end your response with "Thanks for asking!"

# Context:
# {context}

# Your Question: {question}

# Here’s my answer:"""



# custom_rag_prompt = PromptTemplate.from_template(template)

# # Initialize chat history
# chat_history = ChatMessageHistory()

# # Function to retrieve user question embedding using Azure OpenAI
# def get_embedding(question):
#     """
#     Generates an embedding vector for the given user question using Azure OpenAI.

#     """
#     try:
#         response = client.embeddings.create(
#             model="embedding",
#             input=[question]
#         )
#         return response.data[0].embedding
#     except Exception as e:
#         logging.error(f"Error generating embedding: {e}")
#         return None

# def index_search(user_question, user_question_vector):
#     """
#     Performs a search in the Azure Cognitive Search index using the user's question and vector embedding.
#     Retrieves relevant documents based on both semantic and vector search.
#     """
#     search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))

#     try:
#         results = search_client.search(
#             search_text=user_question,
#             top=5,
#             vector_queries=[VectorizedQuery(vector=user_question_vector, k_nearest_neighbors=5, fields="contentVector")],
#             query_type="semantic",
#             semantic_configuration_name="my-semantic-config"
#         )

#         sources = [{'id': doc['id'], 'content': doc['content']} for doc in results if 'content' in doc]
        
#         # Filter relevant sources based on content relevance to the user question
#         relevant_sources = [
#             source for source in sources 
#             if any(keyword in source['content'].lower() for keyword in user_question.lower().split())
#         ]

#         return relevant_sources
    
#     except Exception as e:
#         logging.error(f"Error performing index search: {e}")
#         return []

# def format_docs(docs):
#     """
#      Formats the search result documents into a readable format for output.

#     """
#     if not docs:
#         return "No relevant documents found."

#     formatted_sources = []
#     for doc in docs:
#         if isinstance(doc, dict) and 'content' in doc and 'id' in doc:
#             file_name = "tdpPlan.pdf"  # Explicitly set the correct file name
#             file_path = f"D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/{file_name}"
#             file_link = f"file:///{urllib.parse.quote(file_path)}"
#             doc_link = f"[Open Document ID: {file_name}]({file_link})"
#             formatted_sources.append(f"{doc['content']}\n\nSource: {doc_link}")

#     return "\n\n".join(formatted_sources)

# def generate_answer(context, question):
#     """
#     Generates an answer to the user's question based on the provided context using Azure OpenAI's chat model.

#     """
#     if not context or context == "No relevant documents found.":
#         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

#     try:
#         response = client.chat.completions.create(
#             model="chat",
#             temperature=0.0,
#             messages=[ 
#                 {"role": "system", "content": "Assistant helps users by answering questions based on provided documents."},
#                 {"role": "user", "content": custom_rag_prompt.format(context=context, question=question)} 
#             ],
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         logging.error(f"Error generating answer from LLM: {e}")
#         return None


# def perform_search(user_question):
#     """ 
#     Performs the full pipeline of operations for retrieving an answer to a user's question.
#     This includes checking memory, performing search, formatting documents, and generating a final answer.
#     """
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
#         return "I couldn't generate an embedding for your question."

#     # Perform index search
#     sources = index_search(user_question, user_question_vector)

#     if not sources:
#         return "I don't know. Please ask a question related to the provided documents. Thanks for asking!"

#     formatted_sources = format_docs(sources[:3])  # Format top 3 sources
#     context = formatted_sources
#     question = user_question

#     # Update chat history
#     chat_history.add_message(user_question)

#     # Generate the answer based on the sources only if relevant sources are found
#     if context:
#         answer = generate_answer(context, question)
#         if answer:
#             # Save the result to memory
#             update_memory(user_question, ["tdpPlan.pdf"], answer)
#             chat_history.add_message(answer)

#             logging.info(f"Full Answer: {answer}")
#             return answer
#     else:
#         return "I couldn't find relevant documents to answer your question."

# def handle_follow_up(user_question):
#     """ 
#     Handles follow-up questions by considering the chat history and re-contextualizing the user's question.
#     """
#     # Get the full chat history for context
#     chat_history_messages = get_chat_history()  # Retrieve the full chat history
#     chat_context = "\n".join([f"User: {msg['user_message']}, Assistant: {msg['assistant_response']}" for msg in chat_history_messages])

#     # Combine the chat context with the current question
#     full_context = f"{chat_context}\nUser: {user_question}"

#     # Check if the question is related to the last question
#     last_message = chat_history.messages[-1] if chat_history.messages else None  # Get the last message
#     if last_message and user_question.startswith(last_message['content']):  # Adjust as necessary
#         return perform_search(full_context)
#     else:
#         return perform_search(user_question)

# if __name__ == "__main__":
#     query = "write multiplication of two numbers in python"
#     print(handle_follow_up(query))  # Call the handler for follow-up





##############################################################
##################### Cloud-query.py (new code) #########################

import os
import logging
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from langchain_core.prompts import PromptTemplate
import urllib.parse
from memory import get_from_memory, update_memory, get_chat_history  # Import for chat history
from langchain_community.chat_message_histories import ChatMessageHistory  # Import for chat history

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Azure Search configuration
AZURE_SEARCH_ENDPOINT = "https://gptkb-jcgzeo3krxxra.search.windows.net"
AZURE_SEARCH_KEY = os.getenv('AZURE_SEARCH_KEY')  # Store sensitive data securely
AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-1')

# Initialize OpenAI client for Azure
client = AzureOpenAI(
    azure_endpoint="https://cog-jcgzeo3krxxra.openai.azure.com/",
    api_key="81505dbbd42945189028d9585b80a042",
    api_version="2024-02-15-preview"
)

# Revised prompt template -(Conversational and Engaging)
template = """Let’s find the answer to your question! Here’s what I gathered from the context. If you don’t see a relevant answer or if it’s unrelated, please say: "I don't know. Please ask a question related to the provided documents."
Remember to always end your response with "Thanks for asking!"

Context:
{context}

Your Question: {question}

Here’s my answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

# Initialize chat history
chat_history = ChatMessageHistory()

# Function to retrieve user question embedding using Azure OpenAI
def get_embedding(question):
    """
    Generates an embedding vector for the given user question using Azure OpenAI.

    """
    try:
        response = client.embeddings.create(
            model="embedding",
            input=[question]
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None

def index_search(user_question, user_question_vector):
    """
    Performs a search in the Azure Cognitive Search index using the user's question and vector embedding.
    Retrieves relevant documents based on both semantic and vector search.
    """
    search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))

    try:
        results = search_client.search(
            search_text=user_question,
            top=5,
            vector_queries=[VectorizedQuery(vector=user_question_vector, k_nearest_neighbors=5, fields="contentVector")],
            query_type="semantic",
            semantic_configuration_name="my-semantic-config"
        )

        # Collect documents that match the query and include 'content'
        sources = [{'id': doc['id'], 'content': doc['content']} for doc in results if 'content' in doc]
        
        # Ensure only relevant documents are returned by checking if question keywords appear in document content
        relevant_sources = []
        for source in sources:
            # Check if any keywords from the user question appear in the document content
            if any(keyword in source['content'].lower() for keyword in user_question.lower().split()):
                relevant_sources.append(source)

        if not relevant_sources:
            logging.info("No relevant documents found.")
            return []  # No relevant documents match the user query

        return relevant_sources
    
    except Exception as e:
        logging.error(f"Error performing index search: {e}")
        return []


def format_docs(docs):
    """
     Formats the search result documents into a readable format for output.

    """
    if not docs:
        return "No relevant documents found."

    formatted_sources = []
    for doc in docs:
        if isinstance(doc, dict) and 'content' in doc and 'id' in doc:
            file_name = "tdpPlan.pdf"  # Explicitly set the correct file name
            file_path = f"D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/{file_name}"
            file_link = f"file:///{urllib.parse.quote(file_path)}"
            doc_link = f"[Open Document ID: {file_name}]({file_link})"
            formatted_sources.append(f"{doc['content']}\n\nSource: {doc_link}")

    return "\n\n".join(formatted_sources)

def generate_answer(context, question):
    """
    Generates an answer to the user's question based on the provided context using Azure OpenAI's chat model.

    """
    if not context or context == "No relevant documents found.":
        return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

    try:
        response = client.chat.completions.create(
            model="chat",
            temperature=0.0,
            messages=[ 
                {"role": "system", "content": "Assistant helps users by answering questions based on provided documents."},
                {"role": "user", "content": custom_rag_prompt.format(context=context, question=question)} 
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating answer from LLM: {e}")
        return None


def perform_search(user_question):
    """ 
    Performs the full pipeline of operations for retrieving an answer to a user's question.
    This includes checking memory, performing search, formatting documents, and generating a final answer.
    """
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
        return "I couldn't generate an embedding for your question."

    # Perform index search
    sources = index_search(user_question, user_question_vector)

    if not sources:  # If no relevant sources are found, block any response from the LLM
        logging.info("No relevant sources found. Blocking LLM from generating a response.")
        return "I don't know. Please ask a question related to the provided documents. Thanks for asking!"

    formatted_sources = format_docs(sources[:3])  # Format top 3 sources
    context = formatted_sources
    question = user_question

    # Update chat history
    chat_history.add_message(user_question)

    # Generate the answer only if relevant sources are found
    if context and "No relevant documents found." not in context:  # Ensure context has relevant data
        answer = generate_answer(context, question)
        if answer:
            # Save the result to memory
            update_memory(user_question, ["tdpPlan.pdf"], answer)
            chat_history.add_message(answer)

            logging.info(f"Full Answer: {answer}")
            return answer
    else:
        return "I couldn't find relevant documents to answer your question."

def handle_follow_up(user_question):
    """ 
    Handles follow-up questions by considering the chat history and re-contextualizing the user's question.
    """
    # Get the full chat history for context
    chat_history_messages = get_chat_history()  # Retrieve the full chat history
    chat_context = "\n".join([f"User: {msg['user_message']}, Assistant: {msg['assistant_response']}" for msg in chat_history_messages])

    # Combine the chat context with the current question
    full_context = f"{chat_context}\nUser: {user_question}"

    # Check if the question is related to the last question
    last_message = chat_history.messages[-1] if chat_history.messages else None  # Get the last message
    if last_message and user_question.startswith(last_message['content']):  # Adjust as necessary
        return perform_search(full_context)
    else:
        return perform_search(user_question)

if __name__ == "__main__":
    query = "what is open compliance?"
    print(handle_follow_up(query))  # Call the handler for follow-up
