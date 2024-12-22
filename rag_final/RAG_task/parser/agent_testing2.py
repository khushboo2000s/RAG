from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize the Azure OpenAI model
llm = AzureChatOpenAI(
    api_key='81505dbbd42945189028d9585b80a042',
    azure_endpoint='https://cog-jcgzeo3krxxra.openai.azure.com/',
    api_version='2024-02-15-preview',
    model="gpt-mini",
    temperature=0.1
)

# Define a simple tool as an example (could be replaced with any needed tool)
@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    print(input)
    return input + 2

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Initialize the agent with the tools and the prompt
tools = [magic_function]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Example query
query = "What documents do I need to open a checking account versus a savings account?"

a = agent_executor.invoke({"input": query})
print(a, '--aaaa')
