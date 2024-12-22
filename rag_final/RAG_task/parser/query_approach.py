################################### cloud-query-test.py ###################################
###########################################################################################
##################### last-5 conversation #################################################


import os
import logging
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from langchain_core.prompts import PromptTemplate
import urllib.parse
from memory import get_from_memory, update_memory  # Import only memory-related functions

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Azure Search configuration
AZURE_SEARCH_ENDPOINT = "https://gptkb-jcgzeo3krxxra.search.windows.net"
AZURE_SEARCH_KEY = os.getenv('AZURE_SEARCH_KEY')
AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-1')

# Initialize OpenAI client for Azure
client = AzureOpenAI(
    azure_endpoint="https://cog-jcgzeo3krxxra.openai.azure.com/",
    api_key="81505dbbd42945189028d9585b80a042",
    api_version="2024-02-15-preview"
)

# Define the revised prompt template
template = """Let’s find the answer to your question! Here’s what I gathered from the context. If you don’t see a relevant answer or if it’s unrelated, please say: "I don't know. Please ask a question related to the provided documents."
Remember to always end your response with "Thanks for asking!"

Context:
{context}

Your Question: {question}

Here’s my answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

# Function to retrieve user question embedding using Azure OpenAI
def get_embedding(question):
    """
    Generates an embedding for a given user question using Azure OpenAI.

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
    Performs a search on the Azure Cognitive Search index using the user question and its embedding.

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
    Formats the search results for display, creating document links with relevant content.

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
    Generates an answer using Azure OpenAI's chat completion model based on the document context and user question.

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

def perform_search(user_question, previous_answer=None):
    """ 
    Executes the entire search and answer generation process, caching results in memory.

    """
    logging.info(f"Performing search for: {user_question}")

    # Check if the query is already stored in memory
    cached_result = get_from_memory(user_question)
    
    # If it's a repeat question, regenerate the answer
    if cached_result and previous_answer and cached_result['answer'] == previous_answer:
        logging.info("Regenerating answer for the same question.")
        return cached_result['answer']

    # Get the question embedding
    user_question_vector = get_embedding(user_question)
    if not user_question_vector:
        logging.warning("Failed to get question vector.")
        return "I couldn't generate an embedding for your question."

    # Perform index search
    sources = index_search(user_question, user_question_vector)

    if not sources:  # If no relevant sources are found, block any response from the LLM
        return "I don't know. Please ask a question related to the provided documents. Thanks for asking!"

    formatted_sources = format_docs(sources[:3])  # Format top 3 sources
    context = formatted_sources
    question = user_question

    # Generate the answer based on the sources
    if context and "No relevant documents found." not in context:  # Ensure context has relevant data
        answer = generate_answer(context, question)
        if answer:
            # Save the result to memory
            update_memory(user_question, ["tdpPlan.pdf"], answer)
            logging.info(f"Full Answer: {answer}")
            return answer
    else:
        return "I couldn't find relevant documents to answer your question."

# Function to handle follow-up questions based on previous answers
def get_contextual_answer(user_question):
    """ 
    Handles follow-up questions by using previous answers as context or performs a fresh search.
    
    """
    previous_answer = get_from_memory(user_question)  # Retrieve the previous answer if it exists

    # If the user is asking a follow-up question, use the last answer as context
    if previous_answer:
        return perform_search(user_question, previous_answer['answer'])
    else:
        return perform_search(user_question)

if __name__ == "__main__":
    query = "explain briefly about space complexity?"
    print(get_contextual_answer(query))  # Call the new function for contextual answer
