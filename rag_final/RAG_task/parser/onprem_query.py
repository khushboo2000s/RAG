# import os
# import logging
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# import faiss
# import pickle
# import urllib.parse
# from memory import get_from_memory, update_memory  
# from azure.search.documents import SearchClient
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents.models import VectorizedQuery
# from openai import AzureOpenAI
# from langchain_core.prompts import PromptTemplate
# import urllib.parse


# # Load environment variables
# load_dotenv()

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # On-Premise Configuration
# ON_PREM_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
# FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'faiss_index.index')
# DOCUMENT_MAPPING_PATH = os.getenv('DOCUMENT_MAPPING_PATH', 'document_mapping.pkl')

# # Load FAISS index and document mapping
# with open(FAISS_INDEX_PATH, 'rb') as f:
#     index = faiss.read_index(FAISS_INDEX_PATH) 

# with open(DOCUMENT_MAPPING_PATH, 'rb') as f:
#     document_mapping = pickle.load(f)

# # Log the type and contents of the document mapping for debugging
# logging.info(f"Document mapping type: {type(document_mapping)}")
# logging.info(f"Document mapping contents: {document_mapping}")

# # Initialize Sentence Transformer model for embeddings
# embedding_model = SentenceTransformer(ON_PREM_EMBEDDING_MODEL)

# # Define the prompt template
# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer or if the question is not related to the provided context, say, "I don't know. Please ask a question related to the provided documents."
# Always say "thanks for asking!" at the end of the answer.

# {context}

# Question: {question}

# Helpful Answer:"""

# custom_rag_prompt = template

# # Function to retrieve user question embedding using SentenceTransformer
# def get_embedding(question):
#     try:
#         embedding = embedding_model.encode([question], convert_to_tensor=False)
#         return embedding[0]  # Return the first embedding (for a single question)
#     except Exception as e:
#         logging.error(f"Error generating embedding: {e}")
#         return None

# # Function to perform FAISS index search
# # def index_search(user_question_vector):
# #     try:
# #         # Perform FAISS search (vector search)
# #         logging.info(f"User question vector: {user_question_vector}")  # Log the question vector
# #         distances, indices = index.search(user_question_vector.reshape(1, -1), k=5)
        
# #         # Collect sources if relevant documents are found
# #         sources = []
# #         for idx in indices[0]:
# #             if idx != -1:  # Exclude invalid indices
# #                 doc_id = document_mapping.get(idx, None)
# #                 if doc_id:
# #                     sources.append({'id': doc_id, 'content': document_mapping[doc_id]['content']})

# #         # Implement content relevance check here (similar logic as in your cloud version)
# #         return sources if sources else []  # Return relevant sources if found
    
# #     except Exception as e:
# #         logging.error(f"Error performing FAISS index search: {e}")
# #         return []  # Return an empty list in case of an error


# ###############################################################################
# def index_search(user_question_vector):
#     try:
#         logging.info(f"User question vector: {user_question_vector}")
#         distances, indices = index.search(user_question_vector.reshape(1, -1), k=5)

#         sources = []
#         for idx in indices[0]:
#             if idx != -1:  # Exclude invalid indices
#                 # Retrieve the document using idx
#                 doc = document_mapping.get(idx)
#                 if doc:
#                     sources.append({'id': idx, 'content': doc['content']})

#         return sources if sources else []
#     except Exception as e:
#         logging.error(f"Error performing FAISS index search: {e}")
#         return []


# ##############################################################################






# # Function to format documents for LLM, including the document ID as a clickable link
# def format_docs(docs):
#     if not docs:
#         logging.warning("No documents to format.")
#         return "No relevant documents found."

#     formatted_sources = []
#     for doc in docs:
#         if isinstance(doc, dict) and 'content' in doc and 'id' in doc:
#             # Construct the file path using forward slashes
#             file_name = "tdpPlan.pdf"  # Explicitly set the correct file name
#             file_path = f"D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/{file_name}"
#             file_link = f"file:///{urllib.parse.quote(file_path)}"  # URL encode the path with forward slashes
#             doc_link = f"[Open Document ID: {file_name}]({file_link})"
#             formatted_sources.append(f"{doc['content']}\n\nSource: {doc_link}")

#     return "\n\n".join(formatted_sources)

# # Function to simulate LLM response (you can modify this logic as needed)
# def generate_answer(context, question):
#     if not context or context == "No relevant documents found.":
#         return "I couldn't find relevant documents to answer your question. Please ask a question related to the provided documents."

#     # Simulate the response generation (since no LLM is involved in on-prem)
#     answer = custom_rag_prompt.format(context=context, question=question)
#     return answer

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
#     if user_question_vector is None or len(user_question_vector) == 0:
#         logging.warning("Failed to get question vector.")
#         return None

#     # Perform index search
#     sources = index_search(user_question_vector)

#     # Check if the sources are empty
#     if not sources:
#         logging.info("No relevant sources found from FAISS search.")  # Log when no relevant sources are found
#         return "I don't know. Please ask a question related to the provided documents. Thanks for asking!"

#     # Prepare the context and question for the LLM
#     formatted_sources = format_docs(sources[:3])  # Format top 3 sources
#     context = formatted_sources
#     question = user_question

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




################### new - approach ##################################################
#####################################################################################
# import logging
# from memory import get_from_memory, update_memory, get_chat_history, add_to_chat_history
# from sentence_transformers import SentenceTransformer
# import faiss
# import pickle
# import urllib.parse
# import os

# # Load environment variables
# # FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', r'D:\Genai_project\Retrieval Augmented Generation\rag_final\RAG_task\parser\faiss_index.index')
# # DOCUMENT_MAPPING_PATH = os.getenv('DOCUMENT_MAPPING_PATH', r'D:\Genai_project\Retrieval Augmented Generation\rag_final\RAG_task\parser\document_mapping.pkl')
# # ON_PREM_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# FAISS_INDEX_PATH = 'D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/parser/faiss_index.index'
# DOCUMENT_MAPPING_PATH = 'D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/parser/document_mapping.pkl'
# ON_PREM_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


# logging.basicConfig(level=logging.WARNING)    ##


# # Load FAISS index and document mapping
# def load_faiss_index():
#     try:
#         index = faiss.read_index(FAISS_INDEX_PATH)
#         logging.info(f"Successfully loaded FAISS index from {FAISS_INDEX_PATH}")
#         return index
#     except Exception as e:
#         logging.error(f"Error loading FAISS index: {e}")
#         return None

# def load_document_mapping():
#     try:
#         if not os.path.exists(DOCUMENT_MAPPING_PATH):
#             logging.error(f"Document mapping file does not exist at {DOCUMENT_MAPPING_PATH}")
#             return None
        
#         with open(DOCUMENT_MAPPING_PATH, 'rb') as f:
#             document_mapping = pickle.load(f)
#             logging.info(f"Successfully loaded document mapping from {DOCUMENT_MAPPING_PATH}")
            
#             # Convert the keys to integers if necessary
#             numeric_mapping = {int(k.split('_')[-1]): v for k, v in document_mapping.items()}

#             # Log the document mapping keys after conversion
#             logging.info(f"Loaded document mapping keys: {list(numeric_mapping.keys())}")
            
#             return numeric_mapping

#     except Exception as e:
#         logging.error(f"Error loading document mapping: {e}")
#         return None



# # Initialize the embedding model
# embedding_model = SentenceTransformer(ON_PREM_EMBEDDING_MODEL)

# # Get embedding for the question
# def get_embedding(question):
#     try:
#         embedding = embedding_model.encode([question], convert_to_tensor=False)
#         logging.info("Successfully generated embedding for the question.")
#         return embedding[0]
#     except Exception as e:
#         logging.error(f"Error generating embedding: {e}")
#         return None

# # Perform FAISS index search
# def index_search(index, user_question_vector, document_mapping):
#     try:
#         # Perform search
#         distances, indices = index.search(user_question_vector.reshape(1, -1), k=5)
#         logging.info(f"FAISS returned indices: {indices} and distances: {distances}")

#         sources = []
#         logging.info(f"Document mapping contains {len(document_mapping)} entries.")

#         # Convert indices to a list
#         index_list = indices[0].tolist()

#         for idx in index_list:
#             # Check if the numeric index exists in the document mapping
#             if idx in document_mapping:  # Now using numeric index keys
#                 doc = document_mapping[idx]
#                 if doc:
#                     logging.info(f"Found document: {doc['content'][:100]}")  # Log first 100 chars of content
#                     sources.append({'id': idx, 'content': doc['content']})
#                 else:
#                     logging.warning(f"Document for index {idx} is None or not properly loaded.")
#             else:
#                 logging.error(f"Index {idx} is not found in the document mapping.")

#         if not sources:
#             logging.error("No valid documents found during index search.")

#         return sources
#     except Exception as e:
#         logging.error(f"Error during FAISS index search: {e}")
#         return []

# # Format the documents for display
# def format_docs(docs):
#     if not docs:
#         return "No relevant documents found."
#     formatted_sources = []
#     for doc in docs:
#         if 'content' in doc and 'id' in doc:
#             # Assume document filename for the example
#             file_path = r"D:/Genai_project/Retrieval Augmented Generation/rag_final/RAG_task/data_files/tdpPlan.pdf"
#             file_link = f"file:///{urllib.parse.quote(file_path)}"
#             doc_link = f"[Open Document ID: tdpPlan.pdf]({file_link})"
#             formatted_sources.append(f"{doc['content']}\n\nSource: {doc_link}")
#     return "\n\n".join(formatted_sources)

# # Simulate LLM answer generation
# def generate_answer(context, question):
#     template = """Use the following pieces of context to answer the question at the end.
#     If you don't know the answer or if the question is not related to the provided context, say, "I don't know. Please ask a question related to the provided documents."
#     Always say "thanks for asking!" at the end of the answer.
    
#     {context}
    
#     Question: {question}
    
#     Helpful Answer:"""
#     return template.format(context=context, question=question)

# # Main function to perform search
# def perform_search(user_question):
#     logging.info(f"Performing search for: {user_question}")

#     # Retrieve chat history to see if context is needed
#     chat_history = get_chat_history()

#     # Add previous questions and answers as context to the current query
#     history_context = "\n".join(
#         [f"User: {item['user_message']}\nAssistant: {item['assistant_response']}" for item in chat_history]
#     )

#     # Check memory for cached result
#     cached_result = get_from_memory(user_question)
#     if cached_result:
#         logging.info("Returning cached result from memory.")
#         return cached_result['answer']

#     # Get question embedding
#     user_question_vector = get_embedding(user_question)
#     if user_question_vector is None:
#         return "Failed to retrieve question embedding."

#     # Load FAISS index and document mapping
#     index = load_faiss_index()
#     document_mapping = load_document_mapping()

#     if not index:
#         return "Failed to load FAISS index."
#     if not document_mapping:
#         return "Failed to load document mapping."

#     # Search the index
#     sources = index_search(index, user_question_vector, document_mapping)

#     if not sources:
#         return "I don't know. Please ask a question related to the provided documents. Thanks for asking!"

#     # Format sources and generate the answer
#     formatted_sources = format_docs(sources[:3])
#     context = f"{history_context}\n\n{formatted_sources}"
#     answer = generate_answer(context, user_question)

#     # Save answer to memory
#     update_memory(user_question, ["tdpPlan.pdf"], answer)

#     # Save chat history
#     add_to_chat_history(user_question, answer)

#     logging.info("Search and answer generation complete.")
#     return answer

# # Entry point for querying
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     user_question = input("Please enter your question: ")
#     response = perform_search(user_question)
#     print(response)

###################################################################################



############# TESTING ##############################
# import os
# import logging
# from dotenv import load_dotenv
# from openai import AzureOpenAI
# from langchain_core.prompts import PromptTemplate
# import numpy as np
# import pickle
# import faiss
# from sentence_transformers import SentenceTransformer

# # Load environment variables
# load_dotenv()

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Azure OpenAI configuration
# client = AzureOpenAI(
#     azure_endpoint="https://cog-jcgzeo3krxxra.openai.azure.com/",
#     api_key="81505dbbd42945189028d9585b80a042",
#     api_version="2024-02-15-preview"
# )

# # Load model for embedding generation
# model_name = os.getenv('EMBEDDING_MODEL_NAME_ON_PREM', 'sentence-transformers/all-mpnet-base-v2')
# model = SentenceTransformer(model_name)

# # Load FAISS index and document mapping paths from environment variables
# faiss_index_path = os.getenv('FAISS_INDEX_PATH')
# document_mapping_path = os.getenv('DOCUMENT_MAPPING_PATH')

# # Define the revised prompt template
# template = """Let’s find the answer to your question! Here’s what I gathered from the context. If you don’t see a relevant answer or if it’s unrelated, please say: "I don't know. Please ask a question related to the provided documents."
# Remember to always end your response with "Thanks for asking!"

# Context:
# {context}

# Your Question: {question}

# Here’s my answer:"""

# # Create the PromptTemplate
# custom_prompt_template = PromptTemplate.from_template(template)

# # Load the FAISS index from disk
# def load_faiss_index(index_path):
#     logging.info(f"Loading FAISS index from {index_path}")
#     index = faiss.read_index(index_path)
#     return index

# # Load the document mapping from disk
# def load_document_mapping(mapping_path):
#     logging.info(f"Loading document mapping from {mapping_path}")
#     with open(mapping_path, 'rb') as f:
#         document_mapping = pickle.load(f)
#     return document_mapping

# # Generate embedding for the search query
# def generate_query_embedding(query):
#     logging.info(f"Generating embedding for query: {query}")
#     query_embedding = model.encode([query])
#     return np.array(query_embedding).astype('float32')

# # Search FAISS index for similar documents
# def search_faiss_index(query_embedding, index, top_k=5):
#     logging.info(f"Searching FAISS index for top {top_k} results.")
#     distances, indices = index.search(query_embedding, top_k)
#     return distances, indices

# # Perform a full query against the FAISS index
# def query_onprem_faiss(query, top_k=5):
#     index = load_faiss_index(faiss_index_path)
#     document_mapping = load_document_mapping(document_mapping_path)

#     query_embedding = generate_query_embedding(query)
#     distances, indices = search_faiss_index(query_embedding, index, top_k)

#     results = []
#     for idx, distance in zip(indices[0], distances[0]):
#         if idx != -1:  # -1 means no result found
#             doc_id = list(document_mapping.keys())[idx]
#             doc_info = document_mapping[doc_id]
#             results.append({
#                 "id": doc_id,
#                 "title": doc_info['title'],
#                 "content": doc_info['content'],
#                 "distance": distance
#             })

#     logging.info(f"Found {len(results)} results.")
#     return results

# # Check if the context is relevant
# def is_context_relevant(context, query):
#     # Here you can implement a simple keyword check or a more sophisticated approach
#     return any(keyword.lower() in context.lower() for keyword in query.split())

# # Generate an answer using Azure OpenAI's chat completion model
# def generate_answer_openai(query, context):
#     try:
#         formatted_prompt = custom_prompt_template.format(context=context, question=query)

#         response = client.chat.completions.create(
#             model="chat",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
#                 {"role": "user", "content": formatted_prompt}
#             ],
#             temperature=0.0,
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         logging.error(f"Error generating answer from Azure OpenAI: {e}")
#         return "I couldn't generate an answer at this time."

# # Example usage
# if __name__ == "__main__":
#     query = "write a program in python of sum of three numbers?"

#     results = query_onprem_faiss(query, top_k=3)
#     context = " ".join([result['content'] for result in results])

#     # Check if context is relevant to the query
#     if is_context_relevant(context, query):
#         answer = generate_answer_openai(query, context)
#     else:
#         answer = "I don't know. Please ask a question related to the provided documents."

#     print(f"Answer: {answer}")



###################################### NEW ####################################
########################## I don't know answer(changes [4] - #*) ###################

import os
import logging
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_core.prompts import PromptTemplate
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Azure OpenAI configuration
client = AzureOpenAI(
    azure_endpoint="https://cog-jcgzeo3krxxra.openai.azure.com/",
    api_key="81505dbbd42945189028d9585b80a042",
    api_version="2024-02-15-preview"
)

# Define the constant message for out-of-scope questions   #*
OUT_OF_SCOPE_MESSAGE = "I don't know. Please ask a question related to the provided documents. Thanks for asking!"



# Load model for embedding generation
model_name = os.getenv('EMBEDDING_MODEL_NAME_ON_PREM', 'sentence-transformers/all-mpnet-base-v2')
model = SentenceTransformer(model_name)

# Load FAISS index and document mapping paths from environment variables
faiss_index_path = os.getenv('FAISS_INDEX_PATH')
document_mapping_path = os.getenv('DOCUMENT_MAPPING_PATH')

# Define the revised prompt template
template = """Let’s find the answer to your question! Here’s what I gathered from the context. If you don’t see a relevant answer or if it’s unrelated, please say: "I don't know. Please ask a question related to the provided documents."
Remember to always end your response with "Thanks for asking!"


Context:
{context}

Your Question: {question}

Here’s my answer:"""

# Create the PromptTemplate
custom_prompt_template = PromptTemplate.from_template(template)

# Load the FAISS index from disk
def load_faiss_index(index_path):
    logging.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    return index

# Load the document mapping from disk
def load_document_mapping(mapping_path):
    logging.info(f"Loading document mapping from {mapping_path}")
    with open(mapping_path, 'rb') as f:
        document_mapping = pickle.load(f)
    return document_mapping

# Generate embedding for the search query
def generate_query_embedding(query):
    logging.info(f"Generating embedding for query: {query}")
    query_embedding = model.encode([query])
    return np.array(query_embedding).astype('float32')

# Search FAISS index for similar documents
def search_faiss_index(query_embedding, index, top_k=5):
    logging.info(f"Searching FAISS index for top {top_k} results.")
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices

# Perform a full query against the FAISS index
def query_onprem_faiss(query, top_k=5):
    index = load_faiss_index(faiss_index_path)
    document_mapping = load_document_mapping(document_mapping_path)

    query_embedding = generate_query_embedding(query)
    distances, indices = search_faiss_index(query_embedding, index, top_k)

    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx != -1:  # -1 means no result found
            doc_id = list(document_mapping.keys())[idx]
            doc_info = document_mapping[doc_id]
            results.append({
                "id": doc_id,
                "title": doc_info['title'],
                "content": doc_info['content'],
                "distance": distance
            })

    logging.info(f"Found {len(results)} results.")

    # Log detailed information about each result
    logging.info(f"Number of results found: {len(results)}")
    # for result in results:
    #     logging.info(f"Document ID: {result['id']}, Title: {result['title']}, Content snippet: {result['content'][:100]}")


    return results

# Check if the context is relevant using semantic similarity
def is_context_relevant(context, query, threshold=0.05):     ## 0.5, #0.05
    # Generate embeddings for both context and query
    context_embedding = model.encode([context])
    query_embedding = model.encode([query])

    # Calculate cosine similarity
    similarity = cosine_similarity(context_embedding, query_embedding)[0][0]
    
    logging.info(f"Computed similarity: {similarity}")
    
    # Check if the similarity is above the threshold
    return similarity >= threshold

# Generate an answer using Azure OpenAI's chat completion model
def generate_answer_openai(query, context):

    if not context:                  #*
        return OUT_OF_SCOPE_MESSAGE 

    
    try:
        formatted_prompt = custom_prompt_template.format(context=context, question=query)

        response = client.chat.completions.create(
            model="chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.0,
        )

        # Log the response for debugging
        # logging.info(f"OpenAI response: {response}")
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating answer from Azure OpenAI: {e}")
        # return "I couldn't generate an answer at this time."   #*
        return OUT_OF_SCOPE_MESSAGE     #*

# Example usage
if __name__ == "__main__":
    query = "email of Jonathan Blanchard "



    results = query_onprem_faiss(query, top_k=5)   ##3 
    context = " ".join([result['content'] for result in results])

    # Log the context to see what is being passed
    # logging.info(f"Context retrieved: {context}")

    # Check if context is relevant to the query using semantic similarity
    if is_context_relevant(context, query):
        answer = generate_answer_openai(query, context)
    else:
        # answer = "I don't know. Please ask a question related to the provided documents."     #*
        answer = OUT_OF_SCOPE_MESSAGE       #*

    print(f"Answer: {answer}")




