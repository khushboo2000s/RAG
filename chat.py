from chat_app.cloud.azure_search import azure_index_search
from chat_app.get_embeddings import get_embedding_from_openai
from chat_app.azure_openai import generate_answer_from_openai
from chat_app.prompts.prompt_template import get_prompt_template
from chat_app.utils.search_utils import format_search_results
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to format documents for LLM
def format_docs(docs):
    if not docs:
        logging.warning("No documents to format.")
        return "No relevant documents found."
    
    return "\n\n".join(doc['content'] for doc in docs if isinstance(doc, dict) and 'content' in doc)


def perform_cloud_search(user_question):
    """
    Perform the search and answer generation using cloud-based services.
    """
    # setup_logging()
    logging.info(f"Performing search for: {user_question}")

    # Get the question embedding from OpenAI
    user_question_vector = get_embedding_from_openai(user_question)
    if not user_question_vector:
        logging.warning("Failed to get question vector.")
        return None

    # Perform the Azure search
    sources = azure_index_search(user_question, user_question_vector)
    if not sources:
        logging.warning("No sources returned from index search.")
        return None



    formatted_sources = format_docs(sources[:3])  # Format top 3 sources
    context = formatted_sources
    question = user_question

    # Generate the answer using OpenAI
    answer = generate_answer_from_openai(context, question)

    if answer:
        logging.info(f"Answer generated successfully: {answer}")
    else:
        logging.error("Failed to generate an answer.")
        
    return answer


    # logging.info(f"Answer: {answer}")
    # return answer

if __name__ == "__main__":
    # Example question
    question = "How Iran raises and moves funds in support of terrorism"
    perform_cloud_search(question)
