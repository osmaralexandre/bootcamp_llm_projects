# %%[markdown]
## LangChain

# %%[markdown]
# Imports

# %%
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks import StdOutCallbackHandler

from dotenv import load_dotenv

load_dotenv()

# %%[markdown]
# Initialize ChatGPT API using langchain_openai

# %%
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0.7,
)

response = llm("What is the capital of France?")
print(response)

# %%[markdown]
# Define a prompt template

# %%
template = """
System: You are a helpful assistant that provides translations.

User: Please translate the following English text to French: "{text}"

Assistant:
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)
print(prompt)

# %%[markdown]
# Chain

# %%
callback = StdOutCallbackHandler()

chain = LLMChain(llm=llm, prompt=prompt, callbacks=[callback])

result = chain.run(text="I love programming.")
print(result)

# %%[markdown]
# Inicializa a memória de conversa

# %%
memory = ConversationBufferMemory()

# Create a conversation chain
conversation = ConversationChain(llm=llm, memory=memory)

# Converse with the model
response1 = conversation.predict(input="Who was the USA president in 2020?")
response2 = conversation.predict(
    input="Could you tell me what was my last question?"
)
print(response1)
print(response2)

# %%[markdown]
# Simple agent creation

# %%
prompt_template = """
You are an intelligent assistant. Based on the following text, answer the question:
Text: {context}
Question: {question}
Answer:
"""

# Prompt configuration
prompt = PromptTemplate(
    input_variables=["context", "question"], template=prompt_template
)

# Create the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt)


# Create the agent using the helper function
def create_custom_agent(llm_chain):
    def _call_agent(context, question):
        return llm_chain.run(context=context, question=question)

    return _call_agent


# Instantiate the agent
agent = create_custom_agent(llm_chain)

# Main function for execution
context = """
LangChain is an open-source library for creating applications with language models. 
It facilitates the development of applications that use language models 
to perform complex tasks, such as text analysis, text generation, and more.
"""
question = "What is LangChain?"

# Running the agent
answer = agent(context, question)
print("Answer:", answer)

# %%
