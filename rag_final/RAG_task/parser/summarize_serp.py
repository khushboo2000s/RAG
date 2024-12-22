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
# from langchain.chains.combine_documents.reduce import (
#     acollapse_docs,
#     split_list_of_docs,
# )
# # from serpapi import GoogleSearch
# import serpapi




# token_max = 1000

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
#         [("system", "Write a concise summary of the following:\n\n{context}")]
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
# async def generate_summary(state: SummaryState):
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

# async def collapse_summaries(state: OverallState):
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

# async def generate_final_summary(state: OverallState):
#     response = await reduce_chain.ainvoke(state["collapsed_summaries"])
#     return {"final_summary": response}


# # Create the summary graph without using functools.partial
# def create_summary_graph(llm: AzureChatOpenAI):
#     graph = StateGraph(OverallState)

#     # Add nodes and edges to the graph
#     graph.add_node("generate_summary", generate_summary)
#     graph.add_node("collect_summaries", collect_summaries)
#     graph.add_node("collapse_summaries", collapse_summaries)
#     graph.add_node("generate_final_summary", generate_final_summary)

#     graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
#     graph.add_edge("generate_summary", "collect_summaries")
#     graph.add_conditional_edges("collect_summaries", should_collapse)
#     graph.add_conditional_edges("collapse_summaries", should_collapse)
#     graph.add_edge("generate_final_summary", END)
    
#     return graph.compile()


# # Async document processing function
# async def process_documents_async(app, split_docs):
#     result = app.astream(
#         {"contents": [doc.page_content for doc in split_docs]},
#         {"recursion_limit": 10}
#     )
#     async for step in result:
#         print(list(step.keys()))
#     print(step, '----final step')

# #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Function to fetch data using SerpAPI
# def fetch_search_results(query: str, serp_api_key: str):
#     # Initialize the client with the API key
#     client = serpapi.Client(api_key=serp_api_key)
    
#     # Define the search parameters
#     params = {
#         'engine': 'google',
#         'q': query
#     }
    
#     # Perform the search
#     results = client.search(params)
#     return results
# #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# # Main async function to coordinate everything
# async def main():
#     global map_chain, reduce_chain, llm

#     serp_api_key = ""
#     query = "Indian International Finance System"

#     # Fetch search results using SerpAPI
#     results = fetch_search_results(query, serp_api_key)

#     # For demonstration, just print the fetched results
#     print(results)

#     # You can process the results (e.g., extracting URLs or snippets)
#     # For now, let's assume the 'results' contain URLs, which we will load and summarize
#     urls = [result['link'] for result in results.get('organic_results', [])]

#     # Load documents from the fetched URLs
#     docs = []
#     for url in urls:
#         docs.extend(load_documents(url))  # Using your existing function

#     # Split documents using your existing method
#     split_docs = split_documents(docs)

#     # Configure Azure OpenAI client
#     llm = AzureChatOpenAI(
#         api_key='',
#         azure_endpoint='https://cog-jcgzeo3krxxra.openai.azure.com/',
#         api_version='2024-02-15-preview',
#         model="chat"
#     )

#     # Define map_chain and reduce_chain globally
#     map_chain = create_map_prompt() | llm | StrOutputParser()
#     reduce_chain = create_reduce_prompt() | llm | StrOutputParser()

#     # Create and run the summary graph
#     app = create_summary_graph(llm)
#     await process_documents_async(app, split_docs)


# # To call the main async function
# if __name__ == "__main__":
#     asyncio.run(main())










# import asyncio
# from typing import Annotated, List, Literal, TypedDict
# import operator
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.chains.llm import LLMChain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import AzureChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_core.documents import Document
# from langgraph.constants import Send
# from langgraph.graph import END, START, StateGraph
# from langchain.chains.combine_documents.reduce import (
#     acollapse_docs,
#     split_list_of_docs,
# )
# import serpapi


# token_max = 1000

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


# class SummaryProcessor:
#     def __init__(self, llm: AzureChatOpenAI):
#         self.llm = llm
#         self.map_chain = self.create_map_chain()
#         self.reduce_chain = self.create_reduce_chain()

#     # Create map and reduce chains
#     def create_map_chain(self):
#         map_prompt = ChatPromptTemplate.from_messages(
#             [("system", "Write a concise summary of the following:\n\n{context}")]
#         )
#         return map_prompt | self.llm | StrOutputParser()

#     def create_reduce_chain(self):
#         reduce_template = """
#         The following is a set of summaries:
#         {docs}
#         Take these and distill it into a final, consolidated summary of the main themes.
#         """
#         reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
#         return reduce_prompt | self.llm | StrOutputParser()

#     # Split documents using CharacterTextSplitter
#     def split_documents(self, docs, chunk_size=500, chunk_overlap=100):
#         text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#             chunk_size=chunk_size, chunk_overlap=chunk_overlap
#         )
#         return text_splitter.split_documents(docs)

#     # Get the number of tokens for a list of documents
#     def length_function(self, documents: List[Document]) -> int:
#         return sum(self.llm.get_num_tokens(doc.page_content) for doc in documents)

#     # Processing functions using instance's map_chain and reduce_chain
#     async def generate_summary(self, state: SummaryState):
#         response = await self.map_chain.ainvoke(state["content"])
#         return {"summaries": [response]}

#     def map_summaries(self, state: OverallState):
#         return [
#             Send("generate_summary", {"content": content}) for content in state["contents"]
#         ]

#     def collect_summaries(self, state: OverallState):
#         return {
#             "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
#         }

#     async def collapse_summaries(self, state: OverallState):
#         doc_lists = split_list_of_docs(
#             state["collapsed_summaries"], lambda docs: self.length_function(docs), token_max
#         )
#         results = []
#         for doc_list in doc_lists:
#             results.append(await acollapse_docs(doc_list, self.reduce_chain.ainvoke))
#         return {"collapsed_summaries": results}

#     def should_collapse(self, state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
#         num_tokens = self.length_function(state["collapsed_summaries"])
#         return "collapse_summaries" if num_tokens > token_max else "generate_final_summary"

#     async def generate_final_summary(self, state: OverallState):
#         response = await self.reduce_chain.ainvoke(state["collapsed_summaries"])
#         return {"final_summary": response}

#     # Method to create the summary graph
#     def create_summary_graph(self):
#         graph = StateGraph(OverallState)

#         # Register methods directly as graph nodes
#         graph.add_node("generate_summary", self.generate_summary)
#         graph.add_node("collect_summaries", self.collect_summaries)
#         graph.add_node("collapse_summaries", self.collapse_summaries)
#         graph.add_node("generate_final_summary", self.generate_final_summary)

#         graph.add_conditional_edges(START, self.map_summaries, ["generate_summary"])
#         graph.add_edge("generate_summary", "collect_summaries")
#         graph.add_conditional_edges("collect_summaries", self.should_collapse)
#         graph.add_conditional_edges("collapse_summaries", self.should_collapse)
#         graph.add_edge("generate_final_summary", END)

#         return graph.compile()

# # Function to fetch data using SerpAPI
# def fetch_search_results(query: str, serp_api_key: str):
#     client = serpapi.Client(api_key=serp_api_key)
#     params = {
#         'engine': 'google',
#         'q': query
#     }
#     results = client.search(params)
#     return results

# # Async document processing function
# async def process_documents_async(app, split_docs):
#     result = app.astream(
#         {"contents": [doc.page_content for doc in split_docs]},
#         {"recursion_limit": 5}
#     )
#     async for step in result:
#         print(list(step.keys()))
#     print(step, '----final step')

# # Main async function to coordinate everything
# async def main():
#     serp_api_key = ""
#     query = "Indian International Finance System"

#     # Fetch search results using SerpAPI
#     results = fetch_search_results(query, serp_api_key)
#     print(results)

#     # Extract URLs from search results
#     urls = [result['link'] for result in results.get('organic_results', [])]

#     # Load documents from URLs
#     docs = []
#     for url in urls:
#         docs.extend(load_documents(url))

#     # Configure Azure OpenAI client
#     llm = AzureChatOpenAI(
#         api_key='',
#         azure_endpoint='https://cog-jcgzeo3krxxra.openai.azure.com/',
#         api_version='2024-02-15-preview',
#         model="chat"
#     )

#     # Initialize the processor and graph
#     processor = SummaryProcessor(llm)
#     app = processor.create_summary_graph()

#     # Split documents and process with the graph
#     split_docs = processor.split_documents(docs)
#     await process_documents_async(app, split_docs)

# # To call the main async function
# if __name__ == "__main__":
#     asyncio.run(main())




# import asyncio
# from typing import Annotated, List, Literal, TypedDict
# import operator
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.chains.llm import LLMChain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import AzureChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_core.documents import Document
# from langgraph.constants import Send
# from langgraph.graph import END, START, StateGraph
# from langchain.chains.combine_documents.reduce import (
#     acollapse_docs,
#     split_list_of_docs,
# )
# import serpapi


# token_max = 3500  # Set to stay safely within model’s 4096-token context limit

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


# class SummaryProcessor:
#     def __init__(self, llm: AzureChatOpenAI):
#         self.llm = llm
#         self.map_chain = self.create_map_chain()
#         self.reduce_chain = self.create_reduce_chain()

#     # Create map and reduce chains
#     def create_map_chain(self):
#         map_prompt = ChatPromptTemplate.from_messages(
#             [("system", "Write a concise summary of the following:\n\n{context}")]
#         )
#         return map_prompt | self.llm | StrOutputParser()

#     def create_reduce_chain(self):
#         reduce_template = """
#         The following is a set of summaries:
#         {docs}
#         Take these and distill it into a final, consolidated summary of the main themes.
#         """
#         reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
#         return reduce_prompt | self.llm | StrOutputParser()

#     # Split documents using CharacterTextSplitter with token limit check
#     def split_documents(self, docs, chunk_size=250, chunk_overlap=100):
#         text_splitter = CharacterTextSplitter(
#             chunk_size=chunk_size, chunk_overlap=chunk_overlap
#         )
#         return text_splitter.split_documents(docs)

#     # Get the number of tokens for a list of documents
#     def length_function(self, documents: List[Document]) -> int:
#         return sum(self.llm.get_num_tokens(doc.page_content) for doc in documents)

#     # Processing functions using instance's map_chain and reduce_chain
#     async def generate_summary(self, state: SummaryState):
#         response = await self.map_chain.ainvoke(state["content"])
#         return {"summaries": [response]}

#     def map_summaries(self, state: OverallState):
#         return [
#             Send("generate_summary", {"content": content}) for content in state["contents"]
#         ]

#     def collect_summaries(self, state: OverallState):
#         return {
#             "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
#         }

#     async def collapse_summaries(self, state: OverallState):
#         doc_lists = split_list_of_docs(
#             state["collapsed_summaries"], lambda docs: self.length_function(docs), token_max
#         )
#         results = []
#         for doc_list in doc_lists:
#             results.append(await acollapse_docs(doc_list, self.reduce_chain.ainvoke))
#         return {"collapsed_summaries": results}

#     def should_collapse(self, state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
#         num_tokens = self.length_function(state["collapsed_summaries"])
#         return "collapse_summaries" if num_tokens > token_max else "generate_final_summary"

#     async def generate_final_summary(self, state: OverallState):
#         doc_list = split_list_of_docs(state["collapsed_summaries"], self.length_function, token_max)[0]
#         response = await self.reduce_chain.ainvoke(doc_list)
#         return {"final_summary": response}

#     # Method to create the summary graph
#     def create_summary_graph(self):
#         graph = StateGraph(OverallState)

#         # Register methods directly as graph nodes
#         graph.add_node("generate_summary", self.generate_summary)
#         graph.add_node("collect_summaries", self.collect_summaries)
#         graph.add_node("collapse_summaries", self.collapse_summaries)
#         graph.add_node("generate_final_summary", self.generate_final_summary)

#         graph.add_conditional_edges(START, self.map_summaries, ["generate_summary"])
#         graph.add_edge("generate_summary", "collect_summaries")
#         graph.add_conditional_edges("collect_summaries", self.should_collapse)
#         graph.add_conditional_edges("collapse_summaries", self.should_collapse)
#         graph.add_edge("generate_final_summary", END)

#         return graph.compile()

# # Function to fetch data using SerpAPI
# def fetch_search_results(query: str, serp_api_key: str):
#     client = serpapi.Client(api_key=serp_api_key)
#     params = {
#         'engine': 'google',
#         'q': query
#     }
#     results = client.search(params)
#     return results

# # Async document processing function
# async def process_documents_async(app, split_docs):
#     result = app.astream(
#         {"contents": [doc.page_content for doc in split_docs]},
#         {"recursion_limit": 5}
#     )
#     async for step in result:
#         print(list(step.keys()))
#     print(step, '----final step')

# # Main async function to coordinate everything
# async def main():
#     serp_api_key = ""
#     query = "Indian International Finance System"

#     # Fetch search results using SerpAPI
#     results = fetch_search_results(query, serp_api_key)
#     print(results)

#     # Extract URLs from search results
#     urls = [result['link'] for result in results.get('organic_results', [])]

#     # Load documents from URLs
#     docs = []
#     for url in urls:
#         docs.extend(load_documents(url))

#     # Configure Azure OpenAI client
#     llm = AzureChatOpenAI(
#         api_key='',
#         azure_endpoint='https://cog-jcgzeo3krxxra.openai.azure.com/',
#         api_version='2024-02-15-preview',
#         model="chat"
#     )

#     # Initialize the processor and graph
#     processor = SummaryProcessor(llm)
#     app = processor.create_summary_graph()

#     # Split documents and process with the graph
#     split_docs = processor.split_documents(docs)
#     await process_documents_async(app, split_docs)

# # To call the main async function
# if __name__ == "__main__":
#     asyncio.run(main())




########## previous+serpapi ################################
# import asyncio
# from langchain_community.document_loaders import WebBaseLoader
# from openai import AzureOpenAI
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.llm import LLMChain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import AzureChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain import hub
# from langchain_text_splitters import CharacterTextSplitter
# import operator
# from typing import Annotated, List, Literal, TypedDict
# from langchain.chains.combine_documents.reduce import (
#     acollapse_docs,
#     split_list_of_docs,
# )
# from langchain_core.documents import Document
# from langgraph.constants import Send
# from langgraph.graph import END, START, StateGraph
# import serpapi

# # SerpAPI client setup
# def fetch_search_results(query: str, serp_api_key: str):
#     client = serpapi.Client(api_key=serp_api_key)
#     params = {'engine': 'google', 'q': query}
#     results = client.search(params)
#     return results


# serp_api_key = ""

# # Function to load documents from URLs
# def load_documents(url: str):
#     loader = WebBaseLoader(url)
#     loader.requests_kwargs = {"verify": False}  # 
#     return loader.load()

# # Function to fetch URLs and load documents
# async def fetch_and_load_documents(query: str):
#     # Fetch search results using SerpAPI
#     results = fetch_search_results(query, serp_api_key)

#     # Extract URLs from search results
#     urls = [result['link'] for result in results.get('organic_results', [])]

#     # Load documents from the URLs
#     docs = []
#     for url in urls:
#         docs.extend(load_documents(url))

#     return docs

# # Configure the Azure OpenAI client
# llm = AzureChatOpenAI(
#     api_key='',
#     azure_endpoint='https://cog-jcgzeo3krxxra.openai.azure.com/',
#     api_version='2024-02-15-preview',  # Make sure to use the correct version
#     model="chat"
# )

# map_prompt = ChatPromptTemplate.from_messages(
#     [("system", "Write a concise summary of the following:\\n\\n{context}")]
# )

# map_chain = map_prompt | llm | StrOutputParser()

# map_prompt = hub.pull("rlm/map-prompt")
# reduce_template = """
# The following is a set of summaries:
# {docs}
# Take these and distill it into a final, consolidated summary
# of the main themes.
# """

# reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

# reduce_chain = reduce_prompt | llm | StrOutputParser()

# # Adjusted chunk size to avoid model's token limit
# chunk_size = 500  # This should be less than the model's token limit
# chunk_overlap = 0  # Optional: if you want some overlap between chunks
# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=chunk_size, chunk_overlap=chunk_overlap
# )

# token_max = 4096  # The model's token limit

# def length_function(documents: List[Document]) -> int:
#     """Get number of tokens for input contents."""
#     return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

# # This will be the overall state of the main graph.
# class OverallState(TypedDict):
#     contents: List[str]
#     summaries: Annotated[list, operator.add]
#     collapsed_summaries: List[Document]
#     final_summary: str

# class SummaryState(TypedDict):
#     content: str

# async def generate_summary(state: SummaryState):
#     # Check token count before invoking the chain
#     token_count = llm.get_num_tokens(state["content"])
    
#     if token_count > token_max:
#         # Handle the situation where the token count exceeds the model's limit
#         # You can either truncate the text or split it further
#         state["content"] = state["content"][:token_max]  # Truncating content to fit
    
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

# async def collapse_summaries(state: OverallState):
#     doc_lists = split_list_of_docs(
#         state["collapsed_summaries"], length_function, token_max
#     )
#     results = []
#     for doc_list in doc_lists:
#         token_count = sum(length_function([doc]) for doc in doc_list)
        
#         # If token count exceeds limit, reduce the chunk size or collapse further
#         if token_count > token_max:
#             # Optionally truncate or filter documents further
#             # doc_list = doc_list[:2]  # Example: take only the first few documents
#             doc_list = doc_list[:len(doc_list)//2]  # Truncate to half
#         results.append(await acollapse_docs(doc_list, reduce_chain.ainvoke))

#     return {"collapsed_summaries": results}

# def should_collapse(
#     state: OverallState,
# ) -> Literal["collapse_summaries", "generate_final_summary"]:
#     num_tokens = length_function(state["collapsed_summaries"])
#     if num_tokens > token_max:
#         return "collapse_summaries"
#     else:
#         return "generate_final_summary"

# async def generate_final_summary(state: OverallState):
#     response = await reduce_chain.ainvoke(state["collapsed_summaries"])
#     return {"final_summary": response}

# # Construct the graph
# graph = StateGraph(OverallState)
# graph.add_node("generate_summary", generate_summary)
# graph.add_node("collect_summaries", collect_summaries)
# graph.add_node("collapse_summaries", collapse_summaries)
# graph.add_node("generate_final_summary", generate_final_summary)

# graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
# graph.add_edge("generate_summary", "collect_summaries")
# graph.add_conditional_edges("collect_summaries", should_collapse)
# graph.add_conditional_edges("collapse_summaries", should_collapse)
# graph.add_edge("generate_final_summary", END)

# app = graph.compile()

# async def process_documents_async(app, split_docs):
#     result = app.astream(
#         {"contents": [doc.page_content for doc in split_docs]},
#         {"recursion_limit": 10}
#     )

#     async for step in result:
#         print(list(step.keys()))
#     print(step, '----ssss')

# # Main async function to run everything
# async def main():
#     query = "International Finance System"  # Your query for SerpAPI
#     docs = await fetch_and_load_documents(query)  # Fetch and load documents from SerpAPI

#     # Split the loaded documents into chunks
#     split_docs = text_splitter.split_documents(docs)
#     print(f"Generated {len(split_docs)} documents.")

#     # Filter out docs that exceed the model's token limit
#     # split_docs = [doc for doc in split_docs if llm.get_num_tokens(doc.page_content) <= token_max]
#     # Updated split and filter for long documents
#     split_docs = [
#         doc for doc in text_splitter.split_documents(docs)
#         if llm.get_num_tokens(doc.page_content) <= token_max
#     ]
#     print(f"Generated {len(split_docs)} documents after filtering.")

#     # Process the documents
#     await process_documents_async(app, split_docs)

# # To call the main async function
# if __name__ == "__main__":
#     asyncio.run(main())





import asyncio
import os
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
import serpapi

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


# SerpAPI Search Function
def search_serpapi(query: str):
    client = serpapi.Client(api_key="")
    results = client.search({
        'engine': 'google',
        'q': query
    })
    return results


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

    # Example usage of SerpAPI (optional)
    query = "Indian International Finance System"
    serpapi_results = search_serpapi(query)
    print(serpapi_results)

# To call the main async function
if __name__ == "__main__":
    asyncio.run(main())
