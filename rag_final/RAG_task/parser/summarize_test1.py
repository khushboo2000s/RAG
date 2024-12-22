# # # from langchain.llms import AzureChatOpenAI
# # from langchain.agents import load_tools, initialize_agent
# # from langchain.agents import AgentType
# # from openai import AzureOpenAI
# # from langchain_openai import AzureChatOpenAI

# # # Configure the Azure OpenAI client
# # llm = AzureChatOpenAI(
# #         api_key='',
# #         azure_endpoint='https://cog-jcgzeo3krxxra.openai.azure.com/',
# #         api_version='2024-02-15-preview',
# #         model="chat"
# # )

# # # Load the tools, including SerpAPI
# # tools = load_tools(["serpapi"], llm=llm)

# # # Initialize the agent with "zero-shot-react-description" agent type
# # agent = initialize_agent(
# #     tools, 
# #     llm, 
# #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
# #     verbose=True
# # )

# # # Run the agent with a query
# # response = agent.run("which club is Cristiano Ronaldo playing right now?")
# # print(response)






# from langchain.agents import initialize_agent
# from langchain_openai import AzureChatOpenAI
# from langchain.agents import AgentType
# from langchain.tools import SerpAPIWrapper

# # Configure the Azure OpenAI client
# llm = AzureChatOpenAI(
#     api_key='',
#     azure_endpoint='https://cog-jcgzeo3krxxra.openai.azure.com/',
#     api_version='2024-02-15-preview',
#     model="chat"
# )

# # Initialize the SerpAPI tool with your API key
# serpapi_api_key = ""
# serpapi_tool = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

# # Initialize the agent
# tools = [serpapi_tool]
# agent = initialize_agent(
#     tools,
#     llm,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )

# # Run the agent with a query
# response = agent.run("which club is Cristiano Ronaldo playing right now?")
# print(response)






# import serpapi

# def fetch_search_results(query, api_key):
#     client = serpapi.Client(api_key=api_key)  # Correctly initialize client with API key
#     params = {
#         "q": query,
#         "engine": "google"
#     }
#     results = client.search(params)  # Use search method to execute the query
#     return results

# # Example usage
# serp_api_key = ""
# query = "Cristiano Ronaldo current club"
# results = fetch_search_results(query, serp_api_key)
# print(results)





# import serpapi
# from langchain_openai import AzureChatOpenAI
# from langchain.agents import load_tools, initialize_agent
# from langchain.agents import AgentType

# # Function to fetch search results using SerpAPI
# def fetch_search_results(query, api_key):
#     client = serpapi.Client(api_key=api_key)  # Initialize client with API key
#     params = {
#         "q": query,
#         "engine": "google"
#     }
#     results = client.search(params)  # Execute the search with parameters
#     return results

# # Configure the Azure OpenAI client
# llm = AzureChatOpenAI(
#     api_key='', 
#     azure_endpoint='https://cog-jcgzeo3krxxra.openai.azure.com/', 
#     api_version='2024-02-15-preview', 
#     model="chat",
#     temperature=0.8
# )

# # Set the SerpAPI key
# serp_api_key = ""

# # Load the tools, including SerpAPI, and pass the API key
# tools = load_tools(["serpapi"], llm=llm, serpapi_api_key=serp_api_key)

# # Initialize the agent with "zero-shot-react-description" agent type
# agent = initialize_agent(
#     tools, 
#     llm, 
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
#     verbose=True
# )

# # Run the agent with a query to get results from the SerpAPI
# query = "what is the link of cygnus compliance"
# response = agent.run(query)
# print(response)




import serpapi
from langchain_openai import AzureChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# # Function to fetch search results using SerpAPI
# def fetch_search_results(query, api_key):
#     client = serpapi.Client(api_key=api_key)  # Initialize client with API key
#     params = {
#         "q": query,
#         "engine": "google"
#     }
#     results = client.search(params)  # Execute the search with parameters
#     return results



# Function to fetch search results using SerpAPI
def fetch_search_results(query, api_key):
    search = serpapi.GoogleSearch({
        "q": query,
        "engine": "google",
        "api_key": api_key  # Pass the API key as part of the params
    })
    results = search.get_dict()  # Get the results as a dictionary
    return results



    

# Configure the Azure OpenAI client with higher temperature for more detailed responses
llm = AzureChatOpenAI(
    api_key='', 
    azure_endpoint='https://cog-jcgzeo3krxxra.openai.azure.com/', 
    api_version='2024-02-15-preview', 
    model="chat",
    temperature=1.0  # Set to 1.0 for even more detailed responses
)

# Set the SerpAPI key
serp_api_key = ""

# Define a custom prompt to encourage more detail in responses
# template = """Based on the search results provided:

# {observations}

# Please provide a brief response with key details, focusing on the main points and including the relevant link.
# """

# template = """
# Based on the search results provided:

# {observations}

# Please respond to the user query accordingly:
# - If the query includes phrases like "in brief," "summary," or asks for specific points, provide a focused, concise answer that includes essential information without excess detail.
# - If the query is more open-ended or asks for a comprehensive explanation, provide a thorough, detailed response with relevant background, context, and examples if applicable.

# Ensure your response aligns with the user's intent for either a concise or detailed answer.
# """


##########################
template = """
Consider the search results below and the user's query:

{observations}

Please answer the user's question with a response that matches their intent:
- If the question asks for a quick answer, specific detail, or mentions "brief," respond with a clear and concise answer focusing only on the essentials.
- For questions that are broad, open-ended, or request an explanation, provide a thorough answer that covers context, background, and all necessary details.

Be attentive to the specific phrasing in the user's question to ensure your response directly addresses what they are asking. Tailor the depth of your answer based on the query’s tone and keywords.
"""

###########################



prompt = PromptTemplate(input_variables=["observations"], template=template)

# Create the LLM chain with the custom prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Load the tools, including SerpAPI, and pass the API key
tools = load_tools(["serpapi"], llm=llm, serpapi_api_key=serp_api_key)

# Initialize the agent with "zero-shot-react-description" agent type
agent = initialize_agent(
    tools, 
    llm, 
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)

# Run the agent with a query to get results from the SerpAPI
query = "what is the link of cygnus compliance?"
search_results = fetch_search_results(query, serp_api_key)

# Collect the observations from the search results
observations = "\n".join([result["snippet"] for result in search_results.get("organic_results", [])])

# Generate the detailed response using the observations
response = llm_chain.run(observations=observations)

# Print the detailed response
print(response)




