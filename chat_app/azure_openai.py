from openai import AzureOpenAI
from chat_app.prompts.prompt_template import get_prompt_template
import logging
import os

# Initialize OpenAI client for Azure
azure_openai_client = AzureOpenAI(
    azure_endpoint="",
    api_key="",  # Ensure to use the correct API key from the environment
    api_version=""
)


def generate_answer_from_openai(context, question):
    """
    Generate answer using OpenAI LLM based on context and question.
    """
    try:
        # Get the RAG (Retrieve and Generate) prompt
        rag_prompt_template=get_prompt_template()
        print(context,'---context')
        print()
        print()
        # Format the prompt with the context and question
        formatted_prompt = rag_prompt_template.format(context=context, question=question)
        print(formatted_prompt,'--pppp')
        response = azure_openai_client.chat.completions.create(
            model="chat",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "Assistant helps users by answering questions based on provided documents."},
                {"role": "user", "content":formatted_prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating LLM answer: {e}")
        return None