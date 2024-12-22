from langchain_core.prompts import PromptTemplate


def get_prompt_template():
    """
    Returns the prompt template.
    """
    
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know; don't try to make up an answer.
    Always say 'thanks for asking!' at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    return PromptTemplate.from_template(template)
