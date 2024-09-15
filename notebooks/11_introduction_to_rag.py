# %%[markdown]
## Introductionto RAG

# %%[markdown]
# Imports

# %%
import os
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# %% [markdown]
# Test LLM without RAG


# %%
def oscar(film, year, llm):

    prompt = PromptTemplate(
        input_variables=["film", "year"],
        template="Quantos oscars o filme {film} ganhou em {year}",
    )

    oscar_chain = LLMChain(llm=llm, prompt=prompt)

    response = oscar_chain({"film": film, "year": year})

    return response


# %%
llm = OpenAI(temperature=0.5, model="gpt-3.5-turbo-instruct")

# %%
response = oscar("Oppenheimer", 2024, llm)
print(response["text"])

# %% [markdown]
# RAG

# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

prompt = ChatPromptTemplate.from_template(
    """
    Responda a pergunta com base apenas no contexto: {context}
    Pergunta: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)

url = "https://pt.wikipedia.org/wiki/Oppenheimer_(filme)"

loader = WebBaseLoader(url)

docs = loader.load()

embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()

documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()

retriever_chain = create_retrieval_chain(retriever, document_chain)
response = retriever_chain.invoke(
    {"input": "Quantos oscars o filme Oppenheimer ganhou em 2024"}
)
print(response["answer"])

# %%
