# %%[markdown]
## Agents with LangChain

# %%[markdown]
# Imports

# %%
import os
from langchain_openai import ChatOpenAI
from langchain_community.tools import YouTubeSearchTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub

from dotenv import load_dotenv

load_dotenv()

# %%[markdown]
# Initialize ChatGPT API using langchain_openai

# %%
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
)

# %%[markdown]
# Tool
# %%
youtube_tool = YouTubeSearchTool()
tools = [youtube_tool]

# %%[markdown]
# Define a prompt template

# %%
prompt = hub.pull("hwchase17/openai-tools-agent", api_key=OPENAI_API_KEY)

# %%[markdown]
# Agent

# %%
agent = create_openai_functions_agent(llm, tools, prompt)

# %%[markdown]
# Agent executor

# %%
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# %%
questions = [
    {
        "input": "Search for the top 5 YouTube videos related to investiment in crypto"
    },
]
for question in questions:
    agent_executor.invoke(question)

# %%
