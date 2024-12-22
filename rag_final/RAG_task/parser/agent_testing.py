from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
 

# Define the rephrase query function
def rephrase_query(inputs):
    # Check if inputs is a dictionary
    if isinstance(inputs, str):
        user_query = inputs  # Inputs is the user query directly
        chat_history = []  
    else:
        user_query = inputs.get('input')
        chat_history = inputs.get('chat_history')



    # Build the conversation context from chat history
    conversation_context = ""
    for entry in chat_history:
        conversation_context += f"{entry['role'].capitalize()}: {entry['message']}\n"
   
    # Add the new user query to the conversation
    prompt = f"Based on our previous conversation:\n{conversation_context}Now, rephrase the query: {user_query}"
    return prompt
 
# Initialize the Azure OpenAI model
llm = AzureChatOpenAI(
    api_key='81505dbbd42945189028d9585b80a042',
    azure_endpoint='https://cog-jcgzeo3krxxra.openai.azure.com/',
    api_version='2024-02-15-preview',
    model="gpt-mini",
    temperature=0.1
)
 
# Define tools for the agent
tools = [Tool(name="Rephrase Tool", func=rephrase_query, description="Rephrase the user's query based on previous conversation")]
 
# Initialize the agent
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
 
# Define chat history and user input
chat_history = [
    {"role": "user", "message": "What is the process to open a bank account?"},
    {"role": "bot", "message": "To open a bank account, you'll need to provide some documents..."},
    {"role": "user", "message": "What documents are needed?"}
]
user_query = "Can I get a debit card when I open an account?"
 
# Use agent to rephrase query based on chat history
# rephrased_query = agent.run({"query": user_query, "chat_history": chat_history})
rephrased_query = agent.run({"input": user_query, "chat_history": chat_history})
print(rephrased_query)




# from langchain.agents import initialize_agent, Tool, AgentType
# from langchain_openai import AzureChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
 
# # Define the rephrase query function
# def rephrase_query(chat_history, user_query):
#     # Build the conversation context from chat history
#     conversation_context = ""
#     for entry in chat_history:
#         conversation_context += f"{entry['role'].capitalize()}: {entry['message']}\n"
   
#     # Add the new user query to the conversation
#     prompt = f"Based on our previous conversation:\n{conversation_context}Now, rephrase the query: {user_query}"
#     return prompt
 
# # Initialize the Azure OpenAI model
# llm = AzureChatOpenAI(
#     api_key='81505dbbd42945189028d9585b80a042',
#     azure_endpoint='https://cog-jcgzeo3krxxra.openai.azure.com/',
#     api_version='2024-02-15-preview',
#     model="gpt-mini",
#     temperature=0.1
# )
 
# # Define tools for the agent
# tools = [Tool(name="Rephrase Tool", func=rephrase_query, description="Rephrase the user's query based on previous conversation")]
 
# # Initialize the agent
# agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
 
# # Define chat history and user input
# chat_history = [
#     {"role": "user", "message": "What is the process to open a bank account?"},
#     {"role": "bot", "message": "To open a bank account, you'll need to provide some documents..."},
#     {"role": "user", "message": "What documents are needed?"}
# ]
# user_query = "Can you list the necessary documents for opening an account?"
 
# # Use agent to rephrase query based on chat history
# rephrased_query = agent.run(chat_history=chat_history, user_query=user_query)
# print(rephrased_query)