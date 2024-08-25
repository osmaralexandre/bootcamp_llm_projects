# %%[markdown]
## Agents with LangChain

# %%[markdown]
# Imports

# %%
import os
from langchain_openai import ChatOpenAI
from src.scrapper import get_text_from_url
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_response_from_openai(message):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
    )

    response = llm.invoke(message)

    return response


@tool
def documentation_tool(url: str, question: str) -> str:
    """This tool receives as input the URL from the documentation and the question about documentation that the user want to be answered"""

    context = get_text_from_url(url)

    message = [
        SystemMessage(
            content="You're a helpful programming assistant that must explain programming library documentation to users as simple as possible"
        ),
        HumanMessage(content=f"{context} \n\n {question}"),
    ]

    response = get_response_from_openai(message)

    return response


@tool
def black_formater_tool(path: str) -> str:
    """This tool receives as input a file system path to a python script file and runs black formater to format the file's python code"""
    print(path)
    try:
        os.system(f"black {path}")
        return "Done!"
    except:
        return "Did not work!"


toolkit = [documentation_tool, black_formater_tool]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a programming assistant. Use your tools to answer questions.
            If you do not have a tool to answer the question, say so.

            Return only the answerr.
            """,
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_openai_tools_agent(llm, toolkit, prompt)

agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)

# result = agent_executor.invoke(
#     {
#         "input": "Hello!"
#     }
# )

# result = agent_executor.invoke(
#     {
#         "input": "Eu quero que você formate esse meu código python. Path: src/scrapper.py"
#     }
# )

result = agent_executor.invoke(
    {
        "input": "Quais as métricas padrão que o MLFlow fornece para avaliar um modelo de texto baseado na documentação dessa url aqui: https://mlflow.org/docs/latest/llms/llm-evaluate/index.html"
    }
)
