def format_search_results(docs):
    """
    Formats search documents for LLM consumption.
    """
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(doc['content'] for doc in docs if isinstance(doc, dict) and 'content' in doc)

