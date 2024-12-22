# import asyncio
# import os
# import getpass
# from langchain_community.document_loaders import WebBaseLoader
# from openai import AzureOpenAI
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.llm import LLMChain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import AzureChatOpenAI
# from langchain import hub
# from langchain_core.output_parsers import StrOutputParser
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_core.documents import Document
# from langgraph.constants import Send
# from langgraph.graph import END, START, StateGraph
# from typing import Annotated, List, Literal, TypedDict
# import operator


# # Define the SummaryState using TypedDict
# class SummaryState(TypedDict):
#     content: str


# class OverallState(TypedDict):
#     # Notice here we use the operator.add
#     # This is because we want combine all the summaries we generate
#     # from individual nodes back into one list - this is essentially
#     # the "reduce" part
#     contents: List[str]
#     summaries: Annotated[list, operator.add]
#     collapsed_summaries: List[Document]
#     final_summary: str

# # Function to load web document
# def load_documents(url: str):
#     loader = WebBaseLoader(url)
#     return loader.load()

# # Function to create map prompt
# def create_map_prompt():
#     return ChatPromptTemplate.from_messages(
#         [("system", "Write a concise summary of the following:\\n\\n{context}")]
#     )

# # Function to create reduce prompt
# def create_reduce_prompt():
#     reduce_template = """
#     The following is a set of summaries:
#     {docs}
#     Take these and distill it into a final, consolidated summary
#     of the main themes.
#     """
#     return ChatPromptTemplate([("human", reduce_template)])

# # Function to split documents using CharacterTextSplitter
# def split_documents(docs, chunk_size=1000, chunk_overlap=0):
#     text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=chunk_size, chunk_overlap=chunk_overlap
#     )
#     return text_splitter.split_documents(docs)

# # Function to get the number of tokens for a list of documents
# def length_function(documents: List[Document], llm: AzureChatOpenAI) -> int:
#     return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

# # Function to create the summary graph
# def create_summary_graph(llm: AzureChatOpenAI, map_chain, reduce_chain):
#     graph = StateGraph(OverallState)

#     # Add nodes
#     graph.add_node("generate_summary", generate_summary)
#     graph.add_node("collect_summaries", collect_summaries)
#     graph.add_node("collapse_summaries", collapse_summaries)
#     graph.add_node("generate_final_summary", generate_final_summary)

#     # Add conditional and normal edges
#     graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
#     graph.add_edge("generate_summary", "collect_summaries")
#     graph.add_conditional_edges("collect_summaries", should_collapse)
#     graph.add_conditional_edges("collapse_summaries", should_collapse)
#     graph.add_edge("generate_final_summary", END)

#     app = graph.compile()

#     return app

# # Functions for document processing logic
# async def generate_summary(state: SummaryState, map_chain):
#     response = await map_chain.ainvoke(state["content"])
#     return {"summaries": [response]}

# def map_summaries(state: OverallState):
#     return [
#         Send("generate_summary", {"content": content}) for content in state["contents"]
#     ]

# def collect_summaries(state: OverallState):
#     return {
#         "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
#     }

# async def collapse_summaries(state: OverallState, reduce_chain):
#     doc_lists = split_list_of_docs(
#         state["collapsed_summaries"], length_function, token_max
#     )
#     results = []
#     for doc_list in doc_lists:
#         results.append(await acollapse_docs(doc_list, reduce_chain.ainvoke))
#     return {"collapsed_summaries": results}

# def should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
#     num_tokens = length_function(state["collapsed_summaries"], llm)
#     if num_tokens > token_max:
#         return "collapse_summaries"
#     else:
#         return "generate_final_summary"

# async def generate_final_summary(state: OverallState, reduce_chain):
#     response = await reduce_chain.ainvoke(state["collapsed_summaries"])
#     return {"final_summary": response}

# # Function to execute the async document processing
# async def process_documents_async(app, split_docs):
#     result = app.astream(
#         {"contents": [doc.page_content for doc in split_docs]},
#         {"recursion_limit": 10}
#     )
#     async for step in result:
#         print(list(step.keys()))
#     print(step,'----final step')

# # Main async function to coordinate everything
# async def main():
#     # Load the documents from URL
#     url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
#     docs = load_documents(url)

#     # Split the documents into chunks
#     split_docs = split_documents(docs)

#     # Directly configure Azure OpenAI client
#     llm = AzureChatOpenAI(
#         api_key='',
#         azure_endpoint='https://cog-jcgzeo3krxxra.openai.azure.com/',
#         api_version='2024-02-15-preview',  # Make sure to use the correct version
#         model="chat"
#     )

#     # Create map and reduce chains
#     map_prompt = create_map_prompt()
#     map_chain = map_prompt | llm | StrOutputParser()
#     reduce_prompt = create_reduce_prompt()
#     reduce_chain = reduce_prompt | llm | StrOutputParser()

#     # Create the summary processing graph
#     app = create_summary_graph(llm, map_chain, reduce_chain)

#     # Process the documents asynchronously
#     await process_documents_async(app, split_docs)

# # To call the main async function
# if __name__ == "__main__":
#     asyncio.run(main())





# import asyncio
# from typing import Annotated, List, Literal, TypedDict
# import operator
# from langchain_community.document_loaders import WebBaseLoader
# from openai import AzureOpenAI
# from langchain.chains.llm import LLMChain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import AzureChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_core.documents import Document
# from langgraph.constants import Send
# from langgraph.graph import END, START, StateGraph

# # Define SummaryState and OverallState using TypedDict
# class SummaryState(TypedDict):
#     content: str

# class OverallState(TypedDict):
#     contents: List[str]
#     summaries: Annotated[list, operator.add]
#     collapsed_summaries: List[Document]
#     final_summary: str

# # Load web document function
# def load_documents(url: str):
#     loader = WebBaseLoader(url)
#     return loader.load()

# # Create map and reduce prompts
# def create_map_prompt():
#     return ChatPromptTemplate.from_messages(
#         [("system", "Write a concise summary of the following:\\n\\n{context}")]
#     )

# def create_reduce_prompt():
#     reduce_template = """
#     The following is a set of summaries:
#     {docs}
#     Take these and distill it into a final, consolidated summary of the main themes.
#     """
#     return ChatPromptTemplate([("human", reduce_template)])

# # Split documents using CharacterTextSplitter
# def split_documents(docs, chunk_size=1000, chunk_overlap=0):
#     text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=chunk_size, chunk_overlap=chunk_overlap
#     )
#     return text_splitter.split_documents(docs)

# # Get the number of tokens for a list of documents
# def length_function(documents: List[Document], llm: AzureChatOpenAI) -> int:
#     return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

# # Functions for document processing logic
# async def generate_summary(state: SummaryState, map_chain):
#     response = await map_chain.ainvoke(state["content"])
#     return {"summaries": [response]}

# def map_summaries(state: OverallState):
#     return [
#         Send("generate_summary", {"content": content}) for content in state["contents"]
#     ]

# def collect_summaries(state: OverallState):
#     return {
#         "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
#     }

# async def collapse_summaries(state: OverallState, reduce_chain):
#     doc_lists = split_list_of_docs(
#         state["collapsed_summaries"], length_function, token_max
#     )
#     results = []
#     for doc_list in doc_lists:
#         results.append(await acollapse_docs(doc_list, reduce_chain.ainvoke))
#     return {"collapsed_summaries": results}

# def should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
#     num_tokens = length_function(state["collapsed_summaries"], llm)
#     return "collapse_summaries" if num_tokens > token_max else "generate_final_summary"

# async def generate_final_summary(state: OverallState, reduce_chain):
#     response = await reduce_chain.ainvoke(state["collapsed_summaries"])
#     return {"final_summary": response}



# ###################################################
# # Wrapper functions for each function that needs extra arguments

# async def generate_summary_wrapper(state: SummaryState):
#     return await generate_summary(state, map_chain)

# async def collapse_summaries_wrapper(state: OverallState):
#     return await collapse_summaries(state, reduce_chain)

# async def generate_final_summary_wrapper(state: OverallState):
#     return await generate_final_summary(state, reduce_chain)



# # Create the summary graph without using functools.partial
# def create_summary_graph(llm: AzureChatOpenAI, map_chain, reduce_chain):
#     graph = StateGraph(OverallState)

#     # Use the wrapper functions instead of partials
#     graph.add_node("generate_summary", generate_summary_wrapper)
#     graph.add_node("collect_summaries", collect_summaries)
#     graph.add_node("collapse_summaries", collapse_summaries_wrapper)
#     graph.add_node("generate_final_summary", generate_final_summary_wrapper)

#     graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
#     graph.add_edge("generate_summary", "collect_summaries")
#     graph.add_conditional_edges("collect_summaries", should_collapse)
#     graph.add_conditional_edges("collapse_summaries", should_collapse)
#     graph.add_edge("generate_final_summary", END)
    
#     return graph.compile()

# #################################################################

# # Create the summary graph
# # def create_summary_graph(llm: AzureChatOpenAI, map_chain, reduce_chain):
# #     graph = StateGraph(OverallState)
# #     graph.add_node("generate_summary", generate_summary)
# #     graph.add_node("collect_summaries", collect_summaries)
# #     graph.add_node("collapse_summaries", collapse_summaries)
# #     graph.add_node("generate_final_summary", generate_final_summary)
# #     graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
# #     graph.add_edge("generate_summary", "collect_summaries")
# #     graph.add_conditional_edges("collect_summaries", should_collapse)
# #     graph.add_conditional_edges("collapse_summaries", should_collapse)
# #     graph.add_edge("generate_final_summary", END)
# #     return graph.compile()





# # Async document processing function
# async def process_documents_async(app, split_docs):
#     result = app.astream(
#         {"contents": [doc.page_content for doc in split_docs]},
#         {"recursion_limit": 10}
#     )
#     async for step in result:
#         print(list(step.keys()))
#     print(step, '----final step')

# # Main async function to coordinate everything
# async def main():
#     url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
#     docs = load_documents(url)
#     split_docs = split_documents(docs)

#     # Configure Azure OpenAI client
#     llm = AzureChatOpenAI(
#         api_key='',
#         azure_endpoint='https://cog-jcgzeo3krxxra.openai.azure.com/',
#         api_version='2024-02-15-preview',
#         model="chat"
#     )

#     # Create map and reduce chains
#     map_chain = create_map_prompt() | llm | StrOutputParser()
#     reduce_chain = create_reduce_prompt() | llm | StrOutputParser()

#     # Create and run the summary graph
#     app = create_summary_graph(llm, map_chain, reduce_chain)
#     await process_documents_async(app, split_docs)

# # To call the main async function
# if __name__ == "__main__":
#     asyncio.run(main())



######################## FINAL###############################################

import asyncio
from typing import Annotated, List, Literal, TypedDict
import operator
from langchain_community.document_loaders import WebBaseLoader
from openai import AzureOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)


token_max = 2000 

# Define SummaryState and OverallState using TypedDict
class SummaryState(TypedDict):
    content: str

class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str

# Load web document function
def load_documents(url: str):
    loader = WebBaseLoader(url)
    return loader.load()

# Create map and reduce prompts
def create_map_prompt():
    return ChatPromptTemplate.from_messages(
        [("system", "Write a concise summary of the following:\n\n{context}")]
    )

def create_reduce_prompt():
    reduce_template = """
    The following is a set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary of the main themes.
    """
    return ChatPromptTemplate([("human", reduce_template)])

# Split documents using CharacterTextSplitter
def split_documents(docs, chunk_size=1000, chunk_overlap=0):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)

# Get the number of tokens for a list of documents
def length_function(documents: List[Document], llm: AzureChatOpenAI) -> int:
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

# Functions for document processing logic
async def generate_summary(state: SummaryState):
    response = await map_chain.ainvoke(state["content"])
    return {"summaries": [response]}

def map_summaries(state: OverallState):
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]

def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }

async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, token_max
    )
    results = []
    for doc_list in doc_lists:
        results.append(await acollapse_docs(doc_list, reduce_chain.ainvoke))
    return {"collapsed_summaries": results}

def should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"], llm)
    return "collapse_summaries" if num_tokens > token_max else "generate_final_summary"

async def generate_final_summary(state: OverallState):
    response = await reduce_chain.ainvoke(state["collapsed_summaries"])
    return {"final_summary": response}


# Create the summary graph without using functools.partial
def create_summary_graph(llm: AzureChatOpenAI):
    graph = StateGraph(OverallState)

    # Add nodes and edges to the graph
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)

    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)
    
    return graph.compile()


# Async document processing function
async def process_documents_async(app, split_docs):
    result = app.astream(
        {"contents": [doc.page_content for doc in split_docs]},
        {"recursion_limit": 10}
    )
    async for step in result:
        print(list(step.keys()))
    print(step, '----final step')

# Main async function to coordinate everything
async def main():
    global map_chain, reduce_chain, llm

    url = "https://unacademy.com/content/bank-exam/study-material/indian-international-finance-system/financial-knowledge/"
    docs = load_documents(url)
    split_docs = split_documents(docs)

    # Configure Azure OpenAI client
    llm = AzureChatOpenAI(
        api_key='',
        azure_endpoint='https://cog-jcgzeo3krxxra.openai.azure.com/',
        api_version='2024-02-15-preview',
        model="chat"
    )

    # Define map_chain and reduce_chain globally
    map_chain = create_map_prompt() | llm | StrOutputParser()
    reduce_chain = create_reduce_prompt() | llm | StrOutputParser()

    # Create and run the summary graph
    app = create_summary_graph(llm)
    await process_documents_async(app, split_docs)

# To call the main async function
if __name__ == "__main__":
    asyncio.run(main())