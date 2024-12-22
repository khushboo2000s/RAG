# ############### Query-with-chat-history ############################################
# ####################################################################################


# import os
# import logging
# from dotenv import load_dotenv
# from azure.search.documents import SearchClient
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents.models import VectorizedQuery
# from openai import AzureOpenAI
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# import urllib.parse
# from memory import get_from_memory, update_memory  # Import memory functions

# # Load environment variables
# load_dotenv()

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Azure Search configuration
# AZURE_SEARCH_ENDPOINT = "https://gptkb-jcgzeo3krxxra.search.windows.net"
# AZURE_SEARCH_KEY = "2AlK6WpXT16VPryZYgEnTohsHsnmMAvAj7Wa6Do9dCAzSeCk6q9d"
# AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-1')

# # Initialize OpenAI client for Azure
# client = AzureOpenAI(
#     azure_endpoint="https://cog-jcgzeo3krxxra.openai.azure.com/",
#     api_key="81505dbbd42945189028d9585b80a042",
#     api_version="2024-02-15-preview"
# )

# # Define the prompt template
# # template = """Use the following pieces of context to answer the question at the end.
# # If you don't know the answer or if the question is not related to the provided context, say, "I don't know. Please ask a question related to the provided documents."
# # Always say "thanks for asking!" at the end of the answer.

# # {context}

# # Question: {question}

# # Helpful Answer:"""

# ############## Template-Testing ############################
# # Revised prompt template
# template = """You are an intelligent assistant that provides accurate answers based on the context given. Use the following context to formulate a precise answer to the question below.

# If the question is not related to the provided context or if you are unsure, say: "I don't know. Please ask a question related to the provided documents." Always end your response with "Thanks for asking!"

# Context:
# {context}

# Question: {question}

# Answer:"""
# #####################################################################


# custom_rag_prompt = PromptTemplate.from_template(template)

# # Contextualize the question prompt
# contextualize_q_system_prompt = """Given a chat history and the latest user question \
# which might reference context in the chat history, formulate a standalone question \
# which can be understood without the chat history. Do NOT answer the question, \
# just reformulate it if needed and otherwise return it as is."""

# # contextualize_q_prompt = ChatPromptTemplate.from_messages(
# #     [
# #         {"role": "system", "content": contextualize_q_system_prompt},
# #         {"role": "user", "content": "{history}\n\n{question}"}
# #     ]
# # )

# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         ("user", "{history}\n\n{question}")
#     ]
# )


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

# # Function to perform index search with vector search
# def index_search(user_question, user_question_vector):
#     search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
    
#     try:
#         logging.info(f"User question vector: {user_question_vector}")
#         results = search_client.search(
#             search_text=user_question,
#             top=5,
#             vector_queries=[VectorizedQuery(vector=user_question_vector, k_nearest_neighbors=5, fields="contentVector")],
#             query_type="semantic",
#             semantic_configuration_name="my-semantic-config"
#         )
#         sources = [{'id': doc['id'], 'content': doc['content']} for doc in results if 'content' in doc]

#         # Content relevance filtering
#         relevant_sources = [
#             source for source in sources 
#             if any(keyword in source['content'].lower() for keyword in user_question.lower().split())
#         ]

#         logging.info(f"Relevant search results: {relevant_sources}" if relevant_sources else "No relevant documents found.")
#         return relevant_sources
    
#     except Exception as e:
#         logging.error(f"Error performing index search: {e}")
#         return []

# # Function to format the documents for display
# def format_docs(docs):
#     if not docs:
#         logging.warning("No documents to format.")
#         return "No relevant documents found."

#     formatted_sources = []
#     print('-----formatting----')
#     for doc in docs:
#         if isinstance(doc, dict) and 'content' in doc and 'id' in doc:
#             file_name = "tdpPlan.pdf"
#             file_path = f"D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/{file_name}"
#             file_link = f"file:///{urllib.parse.quote(file_path)}"
#             doc_link = f"[Open Document ID: {file_name}]({file_link})"
#             formatted_sources.append(f"{doc['content']}\n\nSource: {doc_link}")

#     return "\n\n".join(formatted_sources)

# # Function to generate an answer based on the context and question
# # def generate_answer(context, question):
# #     if not context or context == "No relevant documents found.":
# #         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

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


# ###################### Testing #################################
# def generate_answer(context, question, chat_history=None):
#     if not context or context == "No relevant documents found.":
#         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

#     try:
#         # Initialize chat history if it's not provided
#         if chat_history is None:
#             chat_history = []

#         # Build the conversation messages using the chat history
#         messages = [
#             {"role": "system", "content": "Assistant helps users by answering questions based on provided documents."}
#         ]

#         # Add chat history (user questions and model responses)
#         for interaction in chat_history:
#             messages.append({"role": "user", "content": interaction["question"]})
#             messages.append({"role": "assistant", "content": interaction["answer"]})

#         # Add the current question to the messages
#         messages.append({"role": "user", "content": custom_rag_prompt.format(context=context, question=question)})

#         # Generate the response using the model
#         response = client.chat.completions.create(
#             model="chat",
#             temperature=0.1,
#             messages=messages,
#         )

#         # Check if the model returned a response
#         if not response.choices:
#             logging.warning("No choices returned from the model.")
#             return "I'm unable to provide an answer at the moment."

#         # Extract the response from the model
#         answer = response.choices[0].message.content

#         # Optionally update chat history with the current question and answer
#         chat_history.append({"question": question, "answer": answer})

#         return answer

#     except Exception as e:
#         logging.error(f"Error generating answer from LLM: {e}")
#         return None
# ########################################################################


# # Chat history function to retrieve and add previous context
# def get_chat_history():
#     # Fetch all previous queries and results from memory for context
#     memory_data = get_from_memory("chat_history")
#     if memory_data:
#         return memory_data['context']
#     return ""

# def update_chat_history(question, answer, sources):
#     # Update chat history with the latest question, answer, and sources
#     chat_history = get_from_memory("chat_history") or {"context": ""}
#     new_entry = f"Question: {question}\nAnswer: {answer}\nSources: {sources}\n"
#     chat_history["context"] += "\n" + new_entry
#     update_memory("chat_history", [], chat_history)

# # Function to contextualize the user question based on chat history
# def contextualize_question(user_question):
#     chat_history = get_chat_history()
#     if not chat_history:
#         return user_question  # No history, return the question as is

#     # Send the user question and chat history to contextualize
#     try:
#         response = client.chat.completions.create(
#             model="chat",
#             temperature=0.0,
#             messages=[
#                 {"role": "system", "content": contextualize_q_system_prompt},
#                 {"role": "user", "content": contextualize_q_prompt.format(history=chat_history, question=user_question)},
#             ],
#         )
#         reformulated_question = response.choices[0].message.content
#         logging.info(f"Reformulated Question: {reformulated_question}")
#         return reformulated_question
#     except Exception as e:
#         logging.error(f"Error contextualizing question: {e}")
#         return user_question

# # Main function to perform the search and answer generation
# def perform_search(user_question):
#     logging.info(f"Performing search for: {user_question}")

#     # Step 1: Contextualize the question if needed
#     user_question = contextualize_question(user_question)

#     cached_result = get_from_memory(user_question)
#     if cached_result:
#         logging.info("Returning result from memory.")
#         return cached_result['answer']

#     user_question_vector = get_embedding(user_question)
#     if not user_question_vector:
#         logging.warning("Failed to get question vector.")
#         return None

#     sources = index_search(user_question, user_question_vector)
#     if not sources:
#         logging.info("No relevant sources found from index search.")
#         return "I don't know. Please ask a question related to the provided documents. Thanks for asking!"

#     formatted_sources = format_docs(sources[:3])
#     context = get_chat_history() + "\n" + formatted_sources

#     answer = generate_answer(context, user_question)

#     if answer:
#         full_answer = f"{answer}\n\nSource: [Open Document ID: tdpPlan.pdf](file:///{urllib.parse.quote('D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/tdpPlan.pdf')})"
#         update_chat_history(user_question, answer, "tdpPlan.pdf")
#         update_memory(user_question, ["tdpPlan.pdf"], full_answer)

#         logging.info(f"Full Answer: {full_answer}")
#         print(f"Full Answer: {full_answer}")
#         return full_answer
#     else:
#         logging.warning("No answer generated by the model.")
#         return "I'm unable to provide an answer at the moment."

# ###############################################################################
# if __name__ == "__main__":
#     query = "Explain about encoder and decoder stacks."
#     perform_search(query)




############### New - Approach #############################################
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

# # Define the prompt template
# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer or if the question is not related to the provided context, say, "I don't know. Please ask a question related to the provided documents."
# Always say "thanks for asking!" at the end of the answer.

# {context}

# Question: {question}

# Helpful Answer:"""

################### Template - Testing ####################################
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

        sources = [{'id': doc['id'], 'content': doc['content']} for doc in results if 'content' in doc]
        relevant_sources = [
            source for source in sources 
            if any(keyword in source['content'].lower() for keyword in user_question.lower().split())
        ]

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
        return None

    # Perform index search
    sources = index_search(user_question, user_question_vector)

    if not sources:
        return "I don't know. Please ask a question related to the provided documents. Thanks for asking!"

    formatted_sources = format_docs(sources[:3])  # Format top 3 sources
    context = formatted_sources
    question = user_question

    # Update chat history
    chat_history.add_message(user_question)

    # Generate the answer based on the sources
    answer = generate_answer(context, question)

    if answer:
        # Save the result to memory
        update_memory(user_question, ["tdpPlan.pdf"], answer)
        chat_history.add_message(answer)

        logging.info(f"Full Answer: {answer}")
        return answer
    else:
        logging.warning("No answer generated by the model.")
        return "I'm unable to provide an answer at the moment."

# Function to handle follow-up questions
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
    query = "salary of Krista Orcutt?"
    print(handle_follow_up(query))  # Call the handler for follow-up
